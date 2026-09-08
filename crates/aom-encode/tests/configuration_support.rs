use aom_encode::key_frame::{encode_key_frame, KeyFrameConfig, KeyFramePlanes};

#[test]
fn query_accepts_native_still_formats() {
    for depth in [8, 10, 12] {
        for (mono, ss_x, ss_y) in [(true, 1, 1), (false, 1, 1), (false, 1, 0), (false, 0, 0)] {
            for speed in [0, 5, 9] {
                let mut config = KeyFrameConfig::allintra_speed0(65, 67, depth, mono, ss_x, ss_y, 0);
                config.cpu_used = speed;
                config.validate_configuration().expect("gated native format");
            }
        }
    }
}

#[test]
fn query_and_encoder_reject_invalid_config_before_reading_planes() {
    let base = KeyFrameConfig::allintra_speed0(65, 67, 8, false, 1, 1, 32);
    let mut cases = [base; 7];
    cases[0].usage = 0;
    cases[1].cpu_used = 10;
    cases[2].bit_depth = 9;
    cases[3].width = 0;
    cases[4].cq_level = 64;
    cases[5].ss_x = 0;
    cases[6].monochrome = true;
    cases[6].ss_y = 0;
    for config in cases {
        let query = config.validate_configuration().unwrap_err();
        let actual = encode_key_frame(KeyFramePlanes { y: &[], u: &[], v: &[] }, &config).unwrap_err();
        assert_eq!(query, actual);
    }
}
