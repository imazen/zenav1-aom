//! Everything between a bitstream's bytes and a block's decisions — the range
//! coder AND the syntax layers built on it (port of libaom v3.14.1).
//!
//! The range coder proper is [`cdf`] + [`dec`] + [`enc`]: encoder (`od_ec_enc`)
//! produces byte-identical output to C, decoder (`od_ec_dec`) recovers
//! identical symbols. The rest of the module is the syntax that rides it,
//! which is the bulk of the code:
//!
//! - [`partition`] — the whole mode-info symbol layer (see its module doc; it
//!   is far more than partition CDFs), 7.5k lines.
//! - [`header`] — the UNCOMPRESSED frame header writers/readers, 3.2k lines.
//! - [`dv_ref`] — intraBC / MV reference-candidate derivation, 2k lines.
//! - [`lr`] — loop-restoration unit syntax.
//! - [`obu`], [`leb128`], [`rb`], [`wb`] — the OBU envelope and the plain
//!   (non-arithmetic) bit readers/writers the uncompressed header uses.
//! - [`default_cdfs`] — the generated libaom default CDF tables.

pub mod cdf;
pub mod dec;
pub mod default_cdfs;
pub mod dv_ref;
pub mod enc;
pub mod header;
pub mod leb128;
pub mod lr;
pub mod obu;
pub mod partition;
pub mod rb;
pub mod wb;

pub use cdf::{read_symbol, update_cdf, write_symbol};
pub use dec::OdEcDec;
pub use enc::OdEcEnc;
