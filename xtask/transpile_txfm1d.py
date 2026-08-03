#!/usr/bin/env python3
"""Transpile libaom ping-pong-buffer forward 1D transforms (av1_fwd_txfm1d.c)
into bit-exact Rust. Handles ONLY the regular statement forms used by
fdct8/16/32/64 and fadst8/16. fadst4 + identities are hand-ported.

`--lanes` emits 8-column lane-batched (`i32x8`) twins (`<name>_impl`) instead,
as ONE `#[magetypes(define(i32x8), v3, neon, -scalar)]` body per kernel — the
macro generates the per-tier variants (`_v3` / `_neon`), so the same transpiled
source serves AVX2 and NEON. `-scalar` drops the macro's default scalar
fallback: the scalar twin is the ALREADY-TRANSCRIBED port in
`transform/{inv_txfm1d_gen,txfm1d_gen}.rs` (the differential's reference), and
the per-tier `prims` helpers this body calls exist only for vector tokens. Each scalar op maps onto the lane helpers in
`aom-dsp/src/transform/simd/prims.rs`: wrapping add/sub stay `+`/`-` (lane adds
wrap), `-a + b` becomes `b - a` (exact in two's complement), a bare `-a` copy
becomes `negv(t, a)`, `half_btf` becomes the exact-i64 `hb` recipe (hand-written
per tier in `prims.rs`) and `clamp_value` becomes `clampv`. Statement order is preserved verbatim so the
lane kernel IS the scalar kernel, per lane.

Every emitted function is validated byte-for-byte by the differential harness;
this transpiler is a convenience, not a trusted oracle."""
import re, sys

ARGS = sys.argv[1:]
INV = '--inv' in ARGS
LANES = '--lanes' in ARGS
LANES16 = '--lanes16' in ARGS
LANES16F = '--lanes16f' in ARGS
FUNCS = [a for a in ARGS if not a.startswith('--')]  # list of extracted .c files

def translate_operand(tok, m):
    """tok like 'bf0[3]' / '-input[7]' / 'cospi[32]' / '-cospi[16]' -> Rust access, negated flag."""
    neg = tok.startswith('-')
    if neg:
        tok = tok[1:]
    mm = re.match(r'^(\w+)\[(\d+)\]$', tok)
    if not mm:
        raise ValueError(f'operand: {tok!r}')
    arr, idx = mm.group(1), mm.group(2)
    arr = m.get(arr, arr)  # bf0/bf1 -> current buffer; input/cospi pass through
    return f'{arr}[{idx}]', neg

def access(tok, m):
    a, neg = translate_operand(tok, m)
    return f'{a}.wrapping_neg()' if neg else a

def access_lanes(tok, m):
    a, neg = translate_operand(tok, m)
    return f'negv(t, {a})' if neg else a

def translate_rhs(rhs, m, stage):
    rhs = rhs.strip()
    # clamp_value(EXPR, stage_range[stage])  (inverse transforms)
    cm = re.match(r'^clamp_value\((.*), stage_range\[stage\]\)$', rhs)
    if cm:
        inner = translate_rhs(cm.group(1), m, stage)
        if LANES:
            return f'clampv(t, {inner}, stage_range[{stage}])'
        return f'clamp_value({inner}, stage_range[{stage}])'
    # half_btf(w0, in0, w1, in1, cos_bit)
    hb = re.match(r'^half_btf\((.*)\)$', rhs)
    if hb:
        args = [a.strip() for a in hb.group(1).split(',')]
        assert len(args) == 5, args
        w0, in0, w1, in1, _bit = args
        def half_arg(t):  # cospi[k] or -cospi[k] or bf0[k]
            a, neg = translate_operand(t, m)
            return f'-{a}' if neg else a
        if LANES:
            # weights are scalar i32 (splat inside hb); operands are lane values
            return (f'hb(t, {half_arg(w0)}, {access_lanes(in0, m)}, '
                    f'{half_arg(w1)}, {access_lanes(in1, m)}, cos_bit)')
        return (f'half_btf({half_arg(w0)}, {access(in0, m)}, '
                f'{half_arg(w1)}, {access(in1, m)}, cos_bit)')
    # binary: A op B  (op in + -), split on top-level ' + ' / ' - '
    bm = re.match(r'^(-?\w+\[\d+\])\s+([+-])\s+(\w+\[\d+\])$', rhs)
    if bm:
        left, op, right = bm.group(1), bm.group(2), bm.group(3)
        if LANES:
            la, lneg = translate_operand(left, m)
            ra, rneg = translate_operand(right, m)
            assert not rneg
            if lneg:
                # -a + b == b - a (two's complement, wrapping — exact);
                # -a - b would need negv(a) - b, never emitted by libaom.
                assert op == '+', f'unsupported lanes form: {rhs!r}'
                return f'{ra} - {la}'
            return f'{la} {op} {ra}'
        l = access(left, m)
        r = access(right, m)
        fn = 'wrapping_add' if op == '+' else 'wrapping_sub'
        return f'{l}.{fn}({r})'
    # unary copy: X[n] or -X[n]
    um = re.match(r'^-?\w+\[\d+\]$', rhs)
    if um:
        if LANES:
            return access_lanes(rhs, m)
        return access(rhs, m)
    raise ValueError(f'RHS: {rhs!r}')

def transpile(path):
    src = open(path).read()
    src = re.sub(r'//[^\n]*', '', src)  # strip line comments before newline-join
    name = re.search(r'void (av1_\w+)\(', src).group(1)
    size = int(re.search(r'const int32_t size = (\d+);', src).group(1))
    m = {'output': 'out', 'step': 'step', 'input': 'input'}  # array-name -> rust var
    ptr = {}  # bf0/bf1 -> 'out'/'step'/'input'
    body = []
    stage = [0]  # mutable stage counter (incremented on each `stage++`)
    # join statements onto single lines
    text = re.sub(r'\n\s+', ' ', src)
    for stmt in text.split(';'):
        stmt = stmt.strip().strip('}').strip('{').strip()
        if not stmt or stmt.startswith('//'):
            continue
        if stmt == 'stage++':
            stage[0] += 1
            continue
        if re.match(r'(void |const int32_t |int32_t |int8_t |int \b|int stage|stage\b|'
                    r'assert\(|av1_range_check_buf\(|\(void\)|cospi = cospi_arr)', stmt):
            # pointer role assignment?
            pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
            if pa:
                ptr[pa.group(1)] = m[pa.group(2)]
            continue
        pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
        if pa:
            ptr[pa.group(1)] = m[pa.group(2)]
            continue
        asn = re.match(r'^(bf[01])\[(\d+)\] = (.*)$', stmt, re.S)
        if asn:
            dst = ptr[asn.group(1)]
            idx = asn.group(2)
            mm = dict(m); mm['bf0'] = ptr.get('bf0'); mm['bf1'] = ptr.get('bf1')
            rhs = translate_rhs(asn.group(3), mm, stage[0])
            body.append(f'    {dst}[{idx}] = {rhs};')
            continue
        raise ValueError(f'unhandled stmt in {name}: {stmt!r}')
    if LANES:
        lines = [f'/// 8-column lane-batched twin of [`crate::{name}`] (transpiled; per-lane',
                 '/// bit-identical to the scalar kernel on the full `i32` domain — pinned by',
                 '/// the `simd::tests` differential at every token permutation, incl. neon).',
                 '#[magetypes(define(i32x8), v3, neon, -scalar)]',
                 '#[allow(unused_variables)]',
                 f'pub(crate) fn {name}_impl(t: Token, input: &[I32x8<Token>], out: &mut [I32x8<Token>], cos_bit: i32, stage_range: &[i8]) {{',
                 f'    let cospi = cospi_arr(cos_bit);',
                 f'    let mut step = [i32x8::zero(t); {size}];',
                 *body,
                 '}']
    else:
        lines = [f'/// Bit-exact port of libaom `{name}` (transpiled, harness-verified).',
                 '#[allow(unused_variables)]',
                 f'pub fn {name}(input: &[i32], out: &mut [i32], cos_bit: i32, stage_range: &[i8]) {{',
                 f'    let cospi = cospi_arr(cos_bit);',
                 f'    let mut step = [0i32; {size}];',
                 *body,
                 '}']
    return '\n'.join(lines)

def transpile16(path):
    """--lanes16: emit a bd8-only 16-lane i16 (`i16x16`) row/column-pass
    twin of an inverse DCT kernel. Two value domains, tracked statically:

      I16 — the i16-representable values: kernel inputs (the driver pre-clamps
            the gathered column to `clamp_value(_, 16)` at bd8) and every
            `clamp_value(_, stage_range=16)` output. Held in `i16x16` lanes.
      T   — unclamped `half_btf` outputs (bounded by |w|<=2^12, |in|<=2^15 to
            ~2^16.03 < 2^17). Held as exact i32 pairs in unpack order (`P32`).

    Op mapping (each PROVEN bit-identical to the scalar op on its domain —
    see `prims16.rs` for the per-helper proofs):
      half_btf(w0,x,w1,y), x/y I16  -> btf16 (exact widening multiply-add)
      clamp(a+b) both I16           -> sadd16 (saturating add == clamp_value 16)
      clamp(a-b) both I16           -> ssub16;  clamp(-a+b) -> ssub16(b,a)
      clamp over any T operand      -> exact i32 add/sub + pack16 (the
                                       saturating narrow == clamp_value 16)
      I16 operand mixed with T      -> ext16 (sign-extend to P32), then above
    Kernels whose scalar form has any OTHER op (identity multiplies, unclamped
    negs/adds, T-fed half_btf) are REJECTED — run them on the i32 path instead.
    """
    src = open(path).read()
    src = re.sub(r'//[^\n]*', '', src)
    name = re.search(r'void (av1_\w+)\(', src).group(1)
    size = int(re.search(r'const int32_t size = (\d+);', src).group(1))
    m = {'output': 'out', 'step': 'step', 'input': 'input'}
    ptr = {}
    body = []
    stage = [0]
    # --- domain state ---
    dom = {}        # (arr, idx) -> 'I' | T-local-name
    vid = {}        # (arr, idx) -> version int (bumped per write)
    nextvid = [0]
    memo_upk = {}   # (op0_key, op1_key) -> upk local name
    counters = {'u': 0, 'x': 0}
    step_used = [False]

    def key(arr, idx):
        return (arr, idx, vid.get((arr, idx), -1))

    def wrote(arr, idx, domain):
        dom[(arr, idx)] = domain
        nextvid[0] += 1
        vid[(arr, idx)] = nextvid[0]
        if arr == 'step' and domain == 'I':
            step_used[0] = True

    def domain_of(arr, idx):
        if arr == 'input':
            return 'I'
        d = dom.get((arr, idx))
        assert d is not None, f'{name}: read of unset {arr}[{idx}]'
        return d

    def i16_ref(arr, idx):
        assert domain_of(arr, idx) == 'I'
        return f'{arr}[{idx}]'

    def operand(tok, mm):
        """'bf0[3]' (possibly '-'-prefixed upstream, resolved by caller) ->
        (arr, idx) in rust names."""
        r = re.match(r'^(\w+)\[(\d+)\]$', tok)
        assert r, tok
        arr = mm.get(r.group(1), r.group(1))
        return arr, int(r.group(2))

    def wexpr(tok, mm):
        neg = tok.startswith('-')
        t2 = tok[1:] if neg else tok
        r = re.match(r'^cospi\[(\d+)\]$', t2)
        assert r, f'weight: {tok}'
        return f'-cospi[{r.group(1)}]' if neg else f'cospi[{r.group(1)}]'

    text = re.sub(r'\n\s+', ' ', src)
    for stmt in text.split(';'):
        stmt = stmt.strip().strip('}').strip('{').strip()
        if not stmt or stmt.startswith('//'):
            continue
        if stmt == 'stage++':
            stage[0] += 1
            continue
        if re.match(r'(void |const int32_t |int32_t |int8_t |int \b|int stage|stage\b|'
                    r'assert\(|av1_range_check_buf\(|\(void\)|cospi = cospi_arr)', stmt):
            pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
            if pa:
                ptr[pa.group(1)] = m[pa.group(2)]
            continue
        pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
        if pa:
            ptr[pa.group(1)] = m[pa.group(2)]
            continue
        asn = re.match(r'^(bf[01])\[(\d+)\] = (.*)$', stmt, re.S)
        assert asn, f'unhandled stmt in {name}: {stmt!r}'
        dst_arr = ptr[asn.group(1)]
        dst_idx = int(asn.group(2))
        rhs = asn.group(3).strip()
        mm = dict(m)
        mm['bf0'] = ptr.get('bf0')
        mm['bf1'] = ptr.get('bf1')

        cm = re.match(r'^clamp_value\((.*), stage_range\[stage\]\)$', rhs)
        hb_m = re.match(r'^half_btf\((.*)\)$', rhs)
        cp_m = re.fullmatch(r'\w+\[\d+\]', rhs)
        if cm:
            inner = cm.group(1).strip()
            bm = re.match(r'^(-?)(\w+\[\d+\])\s*([+-])\s*(\w+\[\d+\])$', inner)
            assert bm, f'{name}: clamp form {inner!r}'
            lneg, ltok, op, rtok = bm.group(1) == '-', bm.group(2), bm.group(3), bm.group(4)
            (la, li), (ra, ri) = operand(ltok, mm), operand(rtok, mm)
            ld, rd = domain_of(la, li), domain_of(ra, ri)
            # normalize to (first, second, sub?) with '-a + b' -> b - a
            if lneg:
                assert op == '+', f'{name}: -a - b never emitted'
                first, fd, second, sd, sub = (ra, ri), rd, (la, li), ld, True
            else:
                first, fd, second, sd, sub = (la, li), ld, (ra, ri), rd, op == '-'
            if fd == 'I' and sd == 'I':
                fn = 'ssub16' if sub else 'sadd16'
                expr = f'{fn}(t, {first[0]}[{first[1]}], {second[0]}[{second[1]}])'
            else:
                def as_p32(oparr_idx, d):
                    if d == 'I':
                        return f'ext16(t, {oparr_idx[0]}[{oparr_idx[1]}])'
                    return d  # T local name
                pf, ps = as_p32(first, fd), as_p32(second, sd)
                fn = 'psub32' if sub else 'padd32'
                expr = f'pack16(t, {fn}(t, {pf}, {ps}))'
            body.append(f'    {dst_arr}[{dst_idx}] = {expr};')
            wrote(dst_arr, dst_idx, 'I')
        elif hb_m:
            args = [a.strip() for a in hb_m.group(1).split(',')]
            assert len(args) == 5, args
            w0, in0, w1, in1, _ = args
            assert not in0.startswith('-') and not in1.startswith('-')
            (xa, xi), (ya, yi) = operand(in0, mm), operand(in1, mm)
            assert domain_of(xa, xi) == 'I' and domain_of(ya, yi) == 'I', \
                f'{name}: half_btf over non-I16 operand — kernel not i16-safe'
            k = (key(xa, xi), key(ya, yi))
            if k not in memo_upk:
                u = f'u{counters["u"]}'
                counters['u'] += 1
                body.append(f'    let {u} = unpk16(t, {i16_ref(xa, xi)}, {i16_ref(ya, yi)});')
                memo_upk[k] = u
            tl = f'x{counters["x"]}'
            counters['x'] += 1
            body.append(f'    let {tl} = btf16(t, {memo_upk[k]}, {wexpr(w0, mm)}, {wexpr(w1, mm)});')
            wrote(dst_arr, dst_idx, tl)
        elif cp_m:
            sa, si = operand(rhs, mm)
            d = domain_of(sa, si)
            if d == 'I':
                body.append(f'    {dst_arr}[{dst_idx}] = {sa}[{si}];')
                wrote(dst_arr, dst_idx, 'I')
            else:
                wrote(dst_arr, dst_idx, d)  # T copy: alias the local, no code
        else:
            raise AssertionError(f'{name}: op not i16-mappable: {rhs!r} — leave on i32 path')

    n_out = size
    bad = [i for i in range(n_out) if dom.get(('out', i)) != 'I']
    assert not bad, f'{name}: terminal non-I16 out slots {bad} — kernel not i16-safe'
    hdr = [f'/// 16-column i16-lane bd8 twin of [`crate::transform::{name}`] (transpiled).',
           '/// Contract: every input lane is `clamp_value(_, 16)`-bounded (the u8 column',
           '/// pass gathers through the saturating pack) and `stage_range == [16; _]`,',
           '/// `cos_bit == 12` (the bd8 inverse constants). Per-lane bit-identical to the',
           '/// scalar kernel on that domain — pinned by the `simd::tests` differential.',
           '#[magetypes(define(i16x16), v3, neon, -scalar)]',
           '#[allow(unused_variables)]',
           f'pub(crate) fn {name}_i16_impl(t: Token, input: &[I16x16<Token>], out: &mut [I16x16<Token>]) {{',
           '    let cospi = cospi_arr(12);']
    if step_used[0]:
        hdr.append(f'    let mut step = [i16x16::zero(t); {size}];')
    return '\n'.join(hdr + body + ['}'])


def transpile16f(path):
    """--lanes16f: emit a 16-lane i16 (`i16x16`) twin of a FORWARD 1D kernel.

    Unlike the inverse `--lanes16` mode there is only ONE domain. The forward
    kernels carry no `clamp_value` at all, so `xtask/audit_i16_fwd.py` proves
    the narrowing differently: it propagates each value's EXACT linear form and
    reports `M*`, the largest |input| at which every value of the kernel stays
    inside i16. The driver gates on `M*` at runtime, and on that domain

      * every lane add/sub is exact in i16 (the true value fits, so the i16
        wrap and the scalar i32 wrap agree — neither wraps);
      * every `half_btf` is a WIDENING multiply-accumulate into i32
        (|w| <= 2^13, |in| <= 2^15 -> |product| <= 2^28, pair sum <= 2^29),
        rounded and narrowed back to i16 by `fbtf16` — and the narrow is exact
        because `M*` bounds the result inside i16 too.

    Op mapping:
      A + B                        -> a + b            (wrapping i16 lane add)
      -A + B                       -> b - a            (exact in two's complement)
      A / -A                       -> a / negv16(t, a)
      half_btf(w0,A,w1,B,cos_bit)  -> fbtf16(t, unpk16(t,a,b), w0, w1, cos_bit)

    `unpk16` is memoized on a VERSIONED operand key so the two butterflies of a
    rotation pair share one interleave (free on NEON, one `vpunpck` pair on
    AVX2) without ever reusing a stale pairing.
    """
    src = open(path).read()
    src = re.sub(r'//[^\n]*', '', src)
    name = re.search(r'void (av1_\w+)\(', src).group(1)
    size = int(re.search(r'const int32_t size = (\d+);', src).group(1))
    m = {'output': 'out', 'step': 'step', 'input': 'input'}
    ptr = {}
    body = []
    vid = {}
    nextvid = [0]
    memo_upk = {}
    counters = {'u': 0}
    step_used = [False]

    def key(arr, idx):
        return (arr, idx, vid.get((arr, idx), -1))

    def wrote(arr, idx):
        nextvid[0] += 1
        vid[(arr, idx)] = nextvid[0]
        if arr == 'step':
            step_used[0] = True

    def operand(tok, mm):
        r = re.match(r'^(\w+)\[(\d+)\]$', tok)
        assert r, tok
        return mm.get(r.group(1), r.group(1)), int(r.group(2))

    def wexpr(tok, mm):
        neg = tok.startswith('-')
        t2 = tok[1:] if neg else tok
        r = re.match(r'^cospi\[(\d+)\]$', t2)
        assert r, f'weight: {tok}'
        return f'-cospi[{r.group(1)}]' if neg else f'cospi[{r.group(1)}]'

    text = re.sub(r'\n\s+', ' ', src)
    for stmt in text.split(';'):
        stmt = stmt.strip().strip('}').strip('{').strip()
        if not stmt or stmt.startswith('//'):
            continue
        if stmt == 'stage++':
            continue
        if re.match(r'(void |const int32_t |int32_t |int8_t |int \b|int stage|stage\b|'
                    r'assert\(|av1_range_check_buf\(|\(void\)|cospi = cospi_arr)', stmt):
            pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
            if pa:
                ptr[pa.group(1)] = m[pa.group(2)]
            continue
        pa = re.match(r'^(bf[01]) = (output|step|input)$', stmt)
        if pa:
            ptr[pa.group(1)] = m[pa.group(2)]
            continue
        asn = re.match(r'^(bf[01])\[(\d+)\] = (.*)$', stmt, re.S)
        assert asn, f'unhandled stmt in {name}: {stmt!r}'
        dst_arr = ptr[asn.group(1)]
        dst_idx = int(asn.group(2))
        rhs = asn.group(3).strip()
        mm = dict(m)
        mm['bf0'] = ptr.get('bf0')
        mm['bf1'] = ptr.get('bf1')

        hb_m = re.match(r'^half_btf\((.*)\)$', rhs)
        bm = re.match(r'^(-?)(\w+\[\d+\])\s*([+-])\s*(\w+\[\d+\])$', rhs)
        um = re.fullmatch(r'(-?)(\w+\[\d+\])', rhs)
        if hb_m:
            args = [a.strip() for a in hb_m.group(1).split(',')]
            assert len(args) == 5, args
            w0, in0, w1, in1, _ = args
            assert not in0.startswith('-') and not in1.startswith('-')
            (xa, xi), (ya, yi) = operand(in0, mm), operand(in1, mm)
            k = (key(xa, xi), key(ya, yi))
            if k not in memo_upk:
                u = f'u{counters["u"]}'
                counters['u'] += 1
                body.append(f'    let {u} = unpk16(t, {xa}[{xi}], {ya}[{yi}]);')
                memo_upk[k] = u
            body.append(f'    {dst_arr}[{dst_idx}] = fbtf16(t, {memo_upk[k]}, '
                        f'{wexpr(w0, mm)}, {wexpr(w1, mm)}, cos_bit);')
        elif bm:
            lneg, ltok, op, rtok = bm.group(1) == '-', bm.group(2), bm.group(3), bm.group(4)
            (la, li), (ra, ri) = operand(ltok, mm), operand(rtok, mm)
            if lneg:
                assert op == '+', f'{name}: -a - b never emitted'
                body.append(f'    {dst_arr}[{dst_idx}] = {ra}[{ri}] - {la}[{li}];')
            else:
                body.append(f'    {dst_arr}[{dst_idx}] = {la}[{li}] {op} {ra}[{ri}];')
        elif um:
            sa, si = operand(um.group(2), mm)
            if um.group(1) == '-':
                body.append(f'    {dst_arr}[{dst_idx}] = negv16(t, {sa}[{si}]);')
            else:
                body.append(f'    {dst_arr}[{dst_idx}] = {sa}[{si}];')
        else:
            raise AssertionError(f'{name}: op not i16-mappable: {rhs!r} — leave on i32 path')
        wrote(dst_arr, dst_idx)

    hdr = [f'/// 16-lane i16 twin of [`crate::transform::{name}`] (transpiled).',
           '/// Contract: |input lane| <= the `M*` reported for this kernel by',
           '/// `xtask/audit_i16_fwd.py`, which the driver enforces at runtime. On that',
           '/// domain this is per-lane bit-identical to the scalar kernel — pinned by the',
           '/// `lowbd16_fwd::tests` differential at every token permutation.',
           '#[magetypes(define(i16x16), v3, neon, -scalar)]',
           '#[allow(unused_variables)]',
           f'pub(crate) fn {name}_i16_impl(t: Token, input: &[I16x16<Token>], '
           'out: &mut [I16x16<Token>], cos_bit: i32) {',
           '    let cospi = cospi_arr(cos_bit);']
    if step_used[0]:
        hdr.append(f'    let mut step = [i16x16::zero(t); {size}];')
    return '\n'.join(hdr + body + ['}'])


kind = 'inverse' if INV else 'forward'
if LANES16F:
    bodies = [transpile16f(f) for f in FUNCS]
    all_text = '\n'.join(bodies)
    helpers = [h for h in ('fbtf16', 'negv16', 'unpk16') if f'{h}(' in all_text]
    print('//! GENERATED by xtask/transpile_txfm1d.py --lanes16f — do not edit by hand.')
    print('//! 16-lane i16 twins of the scalar 1D FORWARD transform kernels: per-lane')
    print('//! bit-identical to the scalar transcription for every input inside the')
    print('//! per-kernel bound `M*` that `xtask/audit_i16_fwd.py` proves and')
    print('//! [`super::lowbd16_fwd::FWD_I16_MAX_IN`] enforces at runtime. `#[magetypes]`')
    print('//! emits one variant per SIMD tier (`_v3` on x86-64, `_neon` on aarch64) from')
    print('//! this single body; the arch-specific helpers it calls are `#[rite(<tier>)]`')
    print('//! in `super::prims16`, cfg-selected there under one unsuffixed name, so this')
    print('//! file needs no cfg and no tier suffixes. Exactness contract: see')
    print('//! [`super::lowbd16_fwd`]. fadst4 is NOT emitted: its stage-1 values are')
    print('//! `sinpi[j] * x` held unshifted (~2^13 x the input), so it has no i16 form.')
    print('#![allow(clippy::needless_range_loop)]')
    print('use archmage::prelude::*;')
    print('use magetypes::simd::generic::i16x16 as I16x16;')
    print('use crate::transform::cospi::cospi_arr;')
    if helpers:
        print(f'use super::prims16::{{{", ".join(helpers)}}};')
    print()
    for b in bodies:
        print(b)
        print()
elif LANES16:
    bodies = [transpile16(f) for f in FUNCS]
    print('//! GENERATED by xtask/transpile_txfm1d.py --lanes16 — do not edit by hand.')
    print('//! bd8-only 16-lane i16 row/column-pass twins of the scalar 1D inverse DCT')
    print('//! kernels: per-lane bit-identical to the scalar transcription on the audited')
    print('//! bd8 domain (the `super::prims16` helpers reproduce the scalar')
    print('//! clamp/half_btf semantics there). `#[magetypes]` emits one variant per SIMD')
    print('//! tier (`_v3` on x86-64, `_neon` on aarch64) from this single body; the')
    print('//! arch-specific helpers it calls are `#[rite(<tier>)]` in `super::prims16`,')
    print('//! cfg-selected there under one unsuffixed name, so this file needs no cfg and')
    print('//! no tier suffixes. Exactness contract: see [`super::lowbd16`]. Only the')
    print('//! audited i16-safe kernels are emitted (idct4/8/16/32/64); iadst/identity')
    print('//! stay on the i32 path.')
    print('#![allow(clippy::needless_range_loop)]')
    print('use archmage::prelude::*;')
    # `define(i16x16)` aliases the type inside each generated variant's BODY only,
    # so the signature has to name the generic type with the substituted `Token`.
    print('use magetypes::simd::generic::i16x16 as I16x16;')
    print('use crate::transform::cospi::cospi_arr;')
    print('use super::prims16::{btf16, ext16, pack16, padd32, psub32, sadd16, ssub16, unpk16};\n')
    for b in bodies:
        print(b)
        print()
elif LANES:
    bodies = [transpile(f) for f in FUNCS]
    all_text = '\n'.join(bodies)
    helpers = [h for h in ('clampv', 'hb', 'negv') if f'{h}(' in all_text]
    print('//! GENERATED by xtask/transpile_txfm1d.py --lanes — do not edit by hand.')
    print(f'//! 8-column lane-batched twins of the scalar 1D {kind} transforms:')
    print('//! per-lane bit-identical to the scalar transcription on the FULL i32 domain')
    print('//! (the `super::prims` helpers reproduce wrapping-i32 / exact-i64 scalar')
    print('//! semantics). `#[magetypes]` emits one variant per SIMD tier (`_v3` on x86-64,')
    print('//! `_neon` on aarch64) from this single body; the arch-specific helpers it calls')
    print('//! (`hb`) are `#[rite(<tier>)]` in `super::prims`, cfg-selected there under one')
    print('//! unsuffixed name, so this file needs no cfg and no tier suffixes.')
    print('#![allow(clippy::needless_range_loop)]')
    print('use archmage::prelude::*;')
    # `define(i32x8)` aliases the type inside each generated variant's BODY only,
    # so the signature has to name the generic type with the substituted `Token`.
    print('use magetypes::simd::generic::i32x8 as I32x8;')
    print('use crate::transform::cospi::cospi_arr;')
    if helpers:
        print(f'use super::prims::{{{", ".join(helpers)}}};')
    print()
    for b in bodies:
        print(b)
        print()
else:
    print('//! GENERATED by xtask/transpile_txfm1d.py — do not edit by hand.')
    print(f'//! Bit-exact ports of libaom v3.14.1 ping-pong 1D {kind} transforms.')
    print('#![allow(clippy::needless_range_loop)]')
    print('use crate::cospi::cospi_arr;')
    if INV:
        print('use crate::fdct::{half_btf, clamp_value};\n')
    else:
        print('use crate::fdct::half_btf;\n')
    for f in FUNCS:
        print(transpile(f))
        print()
