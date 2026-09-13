fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = std::fs::read(&a[1]).unwrap();
    println!("{:#?}", aom_bench::stream_frame_header(&f));
}
