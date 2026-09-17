use aom_decode::frame::decode_frames;
fn main() {
    let path = std::env::args().nth(1).expect("usage: pixdump <stream.obu> <out.raw>");
    let out = std::env::args().nth(2).unwrap();
    let data = std::fs::read(path).expect("read stream");
    let frames = decode_frames(&data).expect("decode");
    let f = &frames[0];
    let mut buf = Vec::new();
    for px in &f.y { buf.extend_from_slice(&px.to_le_bytes()); }
    for px in &f.u { buf.extend_from_slice(&px.to_le_bytes()); }
    for px in &f.v { buf.extend_from_slice(&px.to_le_bytes()); }
    std::fs::write(&out, &buf).unwrap();
    eprintln!("{}x{} bd{} wrote {}", f.width, f.height, f.bit_depth, out);
}
