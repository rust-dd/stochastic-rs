#[derive(Clone, Copy, Debug, PartialEq)]
pub struct r#EulerArgs {
    pub r#family: u32,
    pub r#components: u32,
    pub r#noises: u32,
    pub r#seed: u32,
    pub r#steps: u32,
    pub r#paths: u32,
    pub r#first_path: u32,
    pub r#increments: u32,
    pub r#n_curves: u32,
    pub r#has_jumps: u32,
    pub r#jump_law: u32,
    pub r#step_first: u32,
    pub r#gamma_law: u32,
    pub r#dt: f32,
    pub r#sqrt_dt: f32,
    pub r#jump_lambda: f32,
    pub r#jump_a: f32,
    pub r#jump_b: f32,
    pub r#jump_c: f32,
    pub r#g1_shape: f32,
    pub r#g1_scale: f32,
    pub r#g1_per: f32,
    pub r#g2_shape: f32,
    pub r#g2_scale: f32,
    pub r#g2_per: f32,
    pub r#has_lift: u32,
    pub r#lift_n: u32,
    pub r#lift_db: f32,
    pub r#lift_fb: f32,
    pub r#lift_x0: f32,
    pub r#hist_slot: u32,
    pub r#series_n: u32,
    pub r#series_live: u32,
    pub r#table_n: u32,
    pub r#program_n: u32,
    pub r#table_u0: f32,
    pub r#x0: [f32; 4],
}

impl metal_oxide::GpuValue for r#EulerArgs {
    const SIZE: std::primitive::usize = 160;
    const ALIGNMENT: std::primitive::usize = 4;
    fn layout() -> metal_oxide::Result<metal_oxide::Layout> {
        Ok(metal_oxide::Layout::record("EulerArgs", vec![
            ("family".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("components".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("noises".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("seed".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("steps".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("paths".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("first_path".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("increments".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("n_curves".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("has_jumps".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("jump_law".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("step_first".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("gamma_law".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("dt".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("sqrt_dt".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("jump_lambda".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("jump_a".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("jump_b".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("jump_c".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g1_shape".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g1_scale".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g1_per".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g2_shape".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g2_scale".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("g2_per".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("has_lift".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("lift_n".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("lift_db".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("lift_fb".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("lift_x0".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("hist_slot".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("series_n".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("series_live".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("table_n".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("program_n".into(), <u32 as metal_oxide::GpuValue>::layout()?),
            ("table_u0".into(), <f32 as metal_oxide::GpuValue>::layout()?),
            ("x0".into(), <[f32; 4] as metal_oxide::GpuValue>::layout()?),
        ])?)
    }
    fn zeroed() -> Self {
        Self {
            r#family: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#components: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#noises: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#seed: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#steps: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#paths: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#first_path: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#increments: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#n_curves: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#has_jumps: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#jump_law: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#step_first: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#gamma_law: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#dt: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#sqrt_dt: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#jump_lambda: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#jump_a: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#jump_b: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#jump_c: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g1_shape: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g1_scale: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g1_per: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g2_shape: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g2_scale: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#g2_per: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#has_lift: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#lift_n: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#lift_db: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#lift_fb: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#lift_x0: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#hist_slot: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#series_n: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#series_live: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#table_n: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#program_n: <u32 as metal_oxide::GpuValue>::zeroed(),
            r#table_u0: <f32 as metal_oxide::GpuValue>::zeroed(),
            r#x0: <[f32; 4] as metal_oxide::GpuValue>::zeroed(),
        }
    }
    fn encode(self, bytes: &mut [u8]) {
        assert_eq!(bytes.len(), Self::SIZE);
        bytes.fill(0);
        <u32 as metal_oxide::GpuValue>::encode(self.r#family, &mut bytes[0..4]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#components, &mut bytes[4..8]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#noises, &mut bytes[8..12]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#seed, &mut bytes[12..16]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#steps, &mut bytes[16..20]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#paths, &mut bytes[20..24]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#first_path, &mut bytes[24..28]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#increments, &mut bytes[28..32]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#n_curves, &mut bytes[32..36]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#has_jumps, &mut bytes[36..40]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#jump_law, &mut bytes[40..44]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#step_first, &mut bytes[44..48]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#gamma_law, &mut bytes[48..52]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#dt, &mut bytes[52..56]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#sqrt_dt, &mut bytes[56..60]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#jump_lambda, &mut bytes[60..64]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#jump_a, &mut bytes[64..68]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#jump_b, &mut bytes[68..72]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#jump_c, &mut bytes[72..76]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g1_shape, &mut bytes[76..80]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g1_scale, &mut bytes[80..84]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g1_per, &mut bytes[84..88]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g2_shape, &mut bytes[88..92]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g2_scale, &mut bytes[92..96]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#g2_per, &mut bytes[96..100]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#has_lift, &mut bytes[100..104]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#lift_n, &mut bytes[104..108]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#lift_db, &mut bytes[108..112]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#lift_fb, &mut bytes[112..116]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#lift_x0, &mut bytes[116..120]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#hist_slot, &mut bytes[120..124]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#series_n, &mut bytes[124..128]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#series_live, &mut bytes[128..132]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#table_n, &mut bytes[132..136]);
        <u32 as metal_oxide::GpuValue>::encode(self.r#program_n, &mut bytes[136..140]);
        <f32 as metal_oxide::GpuValue>::encode(self.r#table_u0, &mut bytes[140..144]);
        <[f32; 4] as metal_oxide::GpuValue>::encode(self.r#x0, &mut bytes[144..160]);
    }
    fn decode(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::SIZE);
        Self {
            r#family: <u32 as metal_oxide::GpuValue>::decode(&bytes[0..4]),
            r#components: <u32 as metal_oxide::GpuValue>::decode(&bytes[4..8]),
            r#noises: <u32 as metal_oxide::GpuValue>::decode(&bytes[8..12]),
            r#seed: <u32 as metal_oxide::GpuValue>::decode(&bytes[12..16]),
            r#steps: <u32 as metal_oxide::GpuValue>::decode(&bytes[16..20]),
            r#paths: <u32 as metal_oxide::GpuValue>::decode(&bytes[20..24]),
            r#first_path: <u32 as metal_oxide::GpuValue>::decode(&bytes[24..28]),
            r#increments: <u32 as metal_oxide::GpuValue>::decode(&bytes[28..32]),
            r#n_curves: <u32 as metal_oxide::GpuValue>::decode(&bytes[32..36]),
            r#has_jumps: <u32 as metal_oxide::GpuValue>::decode(&bytes[36..40]),
            r#jump_law: <u32 as metal_oxide::GpuValue>::decode(&bytes[40..44]),
            r#step_first: <u32 as metal_oxide::GpuValue>::decode(&bytes[44..48]),
            r#gamma_law: <u32 as metal_oxide::GpuValue>::decode(&bytes[48..52]),
            r#dt: <f32 as metal_oxide::GpuValue>::decode(&bytes[52..56]),
            r#sqrt_dt: <f32 as metal_oxide::GpuValue>::decode(&bytes[56..60]),
            r#jump_lambda: <f32 as metal_oxide::GpuValue>::decode(&bytes[60..64]),
            r#jump_a: <f32 as metal_oxide::GpuValue>::decode(&bytes[64..68]),
            r#jump_b: <f32 as metal_oxide::GpuValue>::decode(&bytes[68..72]),
            r#jump_c: <f32 as metal_oxide::GpuValue>::decode(&bytes[72..76]),
            r#g1_shape: <f32 as metal_oxide::GpuValue>::decode(&bytes[76..80]),
            r#g1_scale: <f32 as metal_oxide::GpuValue>::decode(&bytes[80..84]),
            r#g1_per: <f32 as metal_oxide::GpuValue>::decode(&bytes[84..88]),
            r#g2_shape: <f32 as metal_oxide::GpuValue>::decode(&bytes[88..92]),
            r#g2_scale: <f32 as metal_oxide::GpuValue>::decode(&bytes[92..96]),
            r#g2_per: <f32 as metal_oxide::GpuValue>::decode(&bytes[96..100]),
            r#has_lift: <u32 as metal_oxide::GpuValue>::decode(&bytes[100..104]),
            r#lift_n: <u32 as metal_oxide::GpuValue>::decode(&bytes[104..108]),
            r#lift_db: <f32 as metal_oxide::GpuValue>::decode(&bytes[108..112]),
            r#lift_fb: <f32 as metal_oxide::GpuValue>::decode(&bytes[112..116]),
            r#lift_x0: <f32 as metal_oxide::GpuValue>::decode(&bytes[116..120]),
            r#hist_slot: <u32 as metal_oxide::GpuValue>::decode(&bytes[120..124]),
            r#series_n: <u32 as metal_oxide::GpuValue>::decode(&bytes[124..128]),
            r#series_live: <u32 as metal_oxide::GpuValue>::decode(&bytes[128..132]),
            r#table_n: <u32 as metal_oxide::GpuValue>::decode(&bytes[132..136]),
            r#program_n: <u32 as metal_oxide::GpuValue>::decode(&bytes[136..140]),
            r#table_u0: <f32 as metal_oxide::GpuValue>::decode(&bytes[140..144]),
            r#x0: <[f32; 4] as metal_oxide::GpuValue>::decode(&bytes[144..160]),
        }
    }
}

pub struct Kernels<'a> {
    device: &'a metal_oxide::Device,
    module: metal_oxide::Module,
    pipeline_0: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_1: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_2: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_3: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_4: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_5: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_6: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_7: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_8: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_9: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_10: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_11: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_12: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_13: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_14: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_15: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_16: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_17: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_18: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_19: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_20: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_21: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_22: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_23: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_24: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_25: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_26: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_27: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_28: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_29: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_30: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_31: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_32: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_33: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_34: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_35: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_36: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_37: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_38: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_39: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_40: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_41: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_42: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_43: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_44: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_45: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_46: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_47: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_48: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_49: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_50: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_51: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_52: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_53: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_54: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_55: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_56: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_57: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_58: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_59: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_60: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_61: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_62: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_63: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_64: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_65: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_66: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_67: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_68: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_69: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_70: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_71: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_72: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_73: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_74: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_75: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_76: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_77: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_78: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_79: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_80: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_81: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_82: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_83: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_84: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_85: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_86: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_87: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_88: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_89: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_90: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_91: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_92: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_93: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_94: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_95: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_96: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_97: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_98: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_99: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_100: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_101: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_102: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_103: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_104: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_105: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_106: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_107: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_108: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_109: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_110: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_111: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_112: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_113: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_114: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_115: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_116: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_117: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_118: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_119: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_120: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_121: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_122: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_123: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_124: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_125: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_126: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_127: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_128: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_129: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_130: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_131: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_132: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_133: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_134: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_135: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_136: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_137: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_138: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_139: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_140: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_141: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_142: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_143: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_144: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_145: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_146: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_147: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_148: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_149: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_150: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_151: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_152: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_153: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_154: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_155: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_156: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_157: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_158: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_159: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_160: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_161: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_162: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_163: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_164: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_165: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_166: std::cell::OnceCell<metal_oxide::Pipeline>,
    pipeline_167: std::cell::OnceCell<metal_oxide::Pipeline>,
}

#[allow(dead_code)]
pub fn load(device: &metal_oxide::Device, directory: impl std::convert::AsRef<std::path::Path>) -> metal_oxide::Result<Kernels<'_>> {
    metal_oxide_kernels(device, metal_oxide::Module::from_artifact(device, directory)?)
}

#[allow(dead_code)]
pub fn load_bytes<'a>(device: &'a metal_oxide::Device, manifest: &str, metallib: &[u8]) -> metal_oxide::Result<Kernels<'a>> {
    metal_oxide_kernels(device, metal_oxide::Module::from_bytes(device, manifest, metallib)?)
}

fn metal_oxide_kernels(device: &metal_oxide::Device, module: metal_oxide::Module) -> metal_oxide::Result<Kernels<'_>> {
    module.verify_abi("336353149e5f6814f180ad6ab955c1f3bd71487137a2ae03f4a7325781736024")?;
    Ok(Kernels {
        device,
        module,
        pipeline_0: std::cell::OnceCell::new(),
        pipeline_1: std::cell::OnceCell::new(),
        pipeline_2: std::cell::OnceCell::new(),
        pipeline_3: std::cell::OnceCell::new(),
        pipeline_4: std::cell::OnceCell::new(),
        pipeline_5: std::cell::OnceCell::new(),
        pipeline_6: std::cell::OnceCell::new(),
        pipeline_7: std::cell::OnceCell::new(),
        pipeline_8: std::cell::OnceCell::new(),
        pipeline_9: std::cell::OnceCell::new(),
        pipeline_10: std::cell::OnceCell::new(),
        pipeline_11: std::cell::OnceCell::new(),
        pipeline_12: std::cell::OnceCell::new(),
        pipeline_13: std::cell::OnceCell::new(),
        pipeline_14: std::cell::OnceCell::new(),
        pipeline_15: std::cell::OnceCell::new(),
        pipeline_16: std::cell::OnceCell::new(),
        pipeline_17: std::cell::OnceCell::new(),
        pipeline_18: std::cell::OnceCell::new(),
        pipeline_19: std::cell::OnceCell::new(),
        pipeline_20: std::cell::OnceCell::new(),
        pipeline_21: std::cell::OnceCell::new(),
        pipeline_22: std::cell::OnceCell::new(),
        pipeline_23: std::cell::OnceCell::new(),
        pipeline_24: std::cell::OnceCell::new(),
        pipeline_25: std::cell::OnceCell::new(),
        pipeline_26: std::cell::OnceCell::new(),
        pipeline_27: std::cell::OnceCell::new(),
        pipeline_28: std::cell::OnceCell::new(),
        pipeline_29: std::cell::OnceCell::new(),
        pipeline_30: std::cell::OnceCell::new(),
        pipeline_31: std::cell::OnceCell::new(),
        pipeline_32: std::cell::OnceCell::new(),
        pipeline_33: std::cell::OnceCell::new(),
        pipeline_34: std::cell::OnceCell::new(),
        pipeline_35: std::cell::OnceCell::new(),
        pipeline_36: std::cell::OnceCell::new(),
        pipeline_37: std::cell::OnceCell::new(),
        pipeline_38: std::cell::OnceCell::new(),
        pipeline_39: std::cell::OnceCell::new(),
        pipeline_40: std::cell::OnceCell::new(),
        pipeline_41: std::cell::OnceCell::new(),
        pipeline_42: std::cell::OnceCell::new(),
        pipeline_43: std::cell::OnceCell::new(),
        pipeline_44: std::cell::OnceCell::new(),
        pipeline_45: std::cell::OnceCell::new(),
        pipeline_46: std::cell::OnceCell::new(),
        pipeline_47: std::cell::OnceCell::new(),
        pipeline_48: std::cell::OnceCell::new(),
        pipeline_49: std::cell::OnceCell::new(),
        pipeline_50: std::cell::OnceCell::new(),
        pipeline_51: std::cell::OnceCell::new(),
        pipeline_52: std::cell::OnceCell::new(),
        pipeline_53: std::cell::OnceCell::new(),
        pipeline_54: std::cell::OnceCell::new(),
        pipeline_55: std::cell::OnceCell::new(),
        pipeline_56: std::cell::OnceCell::new(),
        pipeline_57: std::cell::OnceCell::new(),
        pipeline_58: std::cell::OnceCell::new(),
        pipeline_59: std::cell::OnceCell::new(),
        pipeline_60: std::cell::OnceCell::new(),
        pipeline_61: std::cell::OnceCell::new(),
        pipeline_62: std::cell::OnceCell::new(),
        pipeline_63: std::cell::OnceCell::new(),
        pipeline_64: std::cell::OnceCell::new(),
        pipeline_65: std::cell::OnceCell::new(),
        pipeline_66: std::cell::OnceCell::new(),
        pipeline_67: std::cell::OnceCell::new(),
        pipeline_68: std::cell::OnceCell::new(),
        pipeline_69: std::cell::OnceCell::new(),
        pipeline_70: std::cell::OnceCell::new(),
        pipeline_71: std::cell::OnceCell::new(),
        pipeline_72: std::cell::OnceCell::new(),
        pipeline_73: std::cell::OnceCell::new(),
        pipeline_74: std::cell::OnceCell::new(),
        pipeline_75: std::cell::OnceCell::new(),
        pipeline_76: std::cell::OnceCell::new(),
        pipeline_77: std::cell::OnceCell::new(),
        pipeline_78: std::cell::OnceCell::new(),
        pipeline_79: std::cell::OnceCell::new(),
        pipeline_80: std::cell::OnceCell::new(),
        pipeline_81: std::cell::OnceCell::new(),
        pipeline_82: std::cell::OnceCell::new(),
        pipeline_83: std::cell::OnceCell::new(),
        pipeline_84: std::cell::OnceCell::new(),
        pipeline_85: std::cell::OnceCell::new(),
        pipeline_86: std::cell::OnceCell::new(),
        pipeline_87: std::cell::OnceCell::new(),
        pipeline_88: std::cell::OnceCell::new(),
        pipeline_89: std::cell::OnceCell::new(),
        pipeline_90: std::cell::OnceCell::new(),
        pipeline_91: std::cell::OnceCell::new(),
        pipeline_92: std::cell::OnceCell::new(),
        pipeline_93: std::cell::OnceCell::new(),
        pipeline_94: std::cell::OnceCell::new(),
        pipeline_95: std::cell::OnceCell::new(),
        pipeline_96: std::cell::OnceCell::new(),
        pipeline_97: std::cell::OnceCell::new(),
        pipeline_98: std::cell::OnceCell::new(),
        pipeline_99: std::cell::OnceCell::new(),
        pipeline_100: std::cell::OnceCell::new(),
        pipeline_101: std::cell::OnceCell::new(),
        pipeline_102: std::cell::OnceCell::new(),
        pipeline_103: std::cell::OnceCell::new(),
        pipeline_104: std::cell::OnceCell::new(),
        pipeline_105: std::cell::OnceCell::new(),
        pipeline_106: std::cell::OnceCell::new(),
        pipeline_107: std::cell::OnceCell::new(),
        pipeline_108: std::cell::OnceCell::new(),
        pipeline_109: std::cell::OnceCell::new(),
        pipeline_110: std::cell::OnceCell::new(),
        pipeline_111: std::cell::OnceCell::new(),
        pipeline_112: std::cell::OnceCell::new(),
        pipeline_113: std::cell::OnceCell::new(),
        pipeline_114: std::cell::OnceCell::new(),
        pipeline_115: std::cell::OnceCell::new(),
        pipeline_116: std::cell::OnceCell::new(),
        pipeline_117: std::cell::OnceCell::new(),
        pipeline_118: std::cell::OnceCell::new(),
        pipeline_119: std::cell::OnceCell::new(),
        pipeline_120: std::cell::OnceCell::new(),
        pipeline_121: std::cell::OnceCell::new(),
        pipeline_122: std::cell::OnceCell::new(),
        pipeline_123: std::cell::OnceCell::new(),
        pipeline_124: std::cell::OnceCell::new(),
        pipeline_125: std::cell::OnceCell::new(),
        pipeline_126: std::cell::OnceCell::new(),
        pipeline_127: std::cell::OnceCell::new(),
        pipeline_128: std::cell::OnceCell::new(),
        pipeline_129: std::cell::OnceCell::new(),
        pipeline_130: std::cell::OnceCell::new(),
        pipeline_131: std::cell::OnceCell::new(),
        pipeline_132: std::cell::OnceCell::new(),
        pipeline_133: std::cell::OnceCell::new(),
        pipeline_134: std::cell::OnceCell::new(),
        pipeline_135: std::cell::OnceCell::new(),
        pipeline_136: std::cell::OnceCell::new(),
        pipeline_137: std::cell::OnceCell::new(),
        pipeline_138: std::cell::OnceCell::new(),
        pipeline_139: std::cell::OnceCell::new(),
        pipeline_140: std::cell::OnceCell::new(),
        pipeline_141: std::cell::OnceCell::new(),
        pipeline_142: std::cell::OnceCell::new(),
        pipeline_143: std::cell::OnceCell::new(),
        pipeline_144: std::cell::OnceCell::new(),
        pipeline_145: std::cell::OnceCell::new(),
        pipeline_146: std::cell::OnceCell::new(),
        pipeline_147: std::cell::OnceCell::new(),
        pipeline_148: std::cell::OnceCell::new(),
        pipeline_149: std::cell::OnceCell::new(),
        pipeline_150: std::cell::OnceCell::new(),
        pipeline_151: std::cell::OnceCell::new(),
        pipeline_152: std::cell::OnceCell::new(),
        pipeline_153: std::cell::OnceCell::new(),
        pipeline_154: std::cell::OnceCell::new(),
        pipeline_155: std::cell::OnceCell::new(),
        pipeline_156: std::cell::OnceCell::new(),
        pipeline_157: std::cell::OnceCell::new(),
        pipeline_158: std::cell::OnceCell::new(),
        pipeline_159: std::cell::OnceCell::new(),
        pipeline_160: std::cell::OnceCell::new(),
        pipeline_161: std::cell::OnceCell::new(),
        pipeline_162: std::cell::OnceCell::new(),
        pipeline_163: std::cell::OnceCell::new(),
        pipeline_164: std::cell::OnceCell::new(),
        pipeline_165: std::cell::OnceCell::new(),
        pipeline_166: std::cell::OnceCell::new(),
        pipeline_167: std::cell::OnceCell::new(),
    })
}

fn metal_oxide_pipeline<'a>(
    device: &metal_oxide::Device,
    module: &metal_oxide::Module,
    slot: &'a std::cell::OnceCell<metal_oxide::Pipeline>,
    name: &str,
) -> metal_oxide::Result<&'a metal_oxide::Pipeline> {
    if let Some(pipeline) = slot.get() {
        return Ok(pipeline);
    }
    let pipeline = metal_oxide::Pipeline::new(device, module, name)?;
    Ok(slot.get_or_init(|| pipeline))
}

#[allow(dead_code)]
impl Kernels<'_> {
    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_0_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_0, "euler_0_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_0_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_0, "euler_0_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_1_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_1, "euler_1_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_1_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_1, "euler_1_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_2_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_2, "euler_2_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_2_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_2, "euler_2_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_3_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_3, "euler_3_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_3_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_3, "euler_3_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_4_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_4, "euler_4_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_4_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_4, "euler_4_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_5_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_5, "euler_5_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_5_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_5, "euler_5_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_6_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_6, "euler_6_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_6_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_6, "euler_6_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_7_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_7, "euler_7_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_7_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_7, "euler_7_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_8_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_8, "euler_8_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_8_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_8, "euler_8_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_9_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_9, "euler_9_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_9_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_9, "euler_9_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_10_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_10, "euler_10_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_10_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_10, "euler_10_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_11_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_11, "euler_11_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_11_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_11, "euler_11_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_12_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_12, "euler_12_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_12_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_12, "euler_12_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_13_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_13, "euler_13_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_13_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_13, "euler_13_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_14_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_14, "euler_14_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_14_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_14, "euler_14_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_15_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_15, "euler_15_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_15_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_15, "euler_15_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_16_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_16, "euler_16_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_16_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_16, "euler_16_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_17_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_17, "euler_17_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_17_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_17, "euler_17_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_18_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_18, "euler_18_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_18_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_18, "euler_18_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_19_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_19, "euler_19_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_19_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_19, "euler_19_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_20_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_20, "euler_20_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_20_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_20, "euler_20_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_21_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_21, "euler_21_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_21_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_21, "euler_21_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_22_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_22, "euler_22_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_22_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_22, "euler_22_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_23_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_23, "euler_23_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_23_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_23, "euler_23_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_24_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_24, "euler_24_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_24_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_24, "euler_24_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_25_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_25, "euler_25_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_25_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_25, "euler_25_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_26_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_26, "euler_26_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_26_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_26, "euler_26_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_27_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_27, "euler_27_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_27_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_27, "euler_27_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_28_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_28, "euler_28_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_28_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_28, "euler_28_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_29_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_29, "euler_29_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_29_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_29, "euler_29_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_30_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_30, "euler_30_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_30_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_30, "euler_30_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_31_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_31, "euler_31_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_31_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_31, "euler_31_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_32_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_32, "euler_32_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_32_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_32, "euler_32_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_33_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_33, "euler_33_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_33_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_33, "euler_33_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_34_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_34, "euler_34_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_34_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_34, "euler_34_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_35_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_35, "euler_35_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_35_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_35, "euler_35_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_36_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_36, "euler_36_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_36_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_36, "euler_36_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_37_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_37, "euler_37_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_37_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_37, "euler_37_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_38_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_38, "euler_38_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_38_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_38, "euler_38_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_39_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_39, "euler_39_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_39_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_39, "euler_39_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_40_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_40, "euler_40_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_40_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_40, "euler_40_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_41_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_41, "euler_41_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_41_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_41, "euler_41_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_42_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_42, "euler_42_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_42_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_42, "euler_42_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_43_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_43, "euler_43_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_43_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_43, "euler_43_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_44_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_44, "euler_44_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_44_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_44, "euler_44_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_45_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_45, "euler_45_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_45_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_45, "euler_45_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_46_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_46, "euler_46_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_46_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_46, "euler_46_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_47_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_47, "euler_47_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_47_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_47, "euler_47_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_48_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_48, "euler_48_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_48_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_48, "euler_48_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_49_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_49, "euler_49_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_49_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_49, "euler_49_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_50_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_50, "euler_50_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_50_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_50, "euler_50_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_51_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_51, "euler_51_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_51_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_51, "euler_51_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_52_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_52, "euler_52_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_52_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_52, "euler_52_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_53_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_53, "euler_53_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_53_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_53, "euler_53_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_54_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_54, "euler_54_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_54_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_54, "euler_54_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_55_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_55, "euler_55_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_55_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_55, "euler_55_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_56_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_56, "euler_56_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_56_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_56, "euler_56_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_57_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_57, "euler_57_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_57_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_57, "euler_57_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_58_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_58, "euler_58_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_58_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_58, "euler_58_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_58_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_59, "euler_58_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_58_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_59, "euler_58_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_59_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_60, "euler_59_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_59_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_60, "euler_59_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_59_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_61, "euler_59_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_59_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_61, "euler_59_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_60_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_62, "euler_60_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_60_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_62, "euler_60_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_60_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_63, "euler_60_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_60_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_63, "euler_60_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_61_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_64, "euler_61_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_61_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_64, "euler_61_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_62_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_65, "euler_62_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_62_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_65, "euler_62_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_62_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_66, "euler_62_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_62_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_66, "euler_62_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_63_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_67, "euler_63_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_63_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_67, "euler_63_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_64_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_68, "euler_64_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_64_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_68, "euler_64_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_65_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_69, "euler_65_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_65_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_69, "euler_65_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_66_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_70, "euler_66_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_66_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_70, "euler_66_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_66_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_71, "euler_66_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_66_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_71, "euler_66_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_67_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_72, "euler_67_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_67_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_72, "euler_67_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_67_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_73, "euler_67_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_67_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_73, "euler_67_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_68_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_74, "euler_68_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_68_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_74, "euler_68_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_68_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_75, "euler_68_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_68_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_75, "euler_68_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_69_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_76, "euler_69_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_69_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_76, "euler_69_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_70_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_77, "euler_70_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_70_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_77, "euler_70_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_71_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_78, "euler_71_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_71_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_78, "euler_71_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_72_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_79, "euler_72_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_72_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_79, "euler_72_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_73_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_80, "euler_73_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_73_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_80, "euler_73_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_74_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_81, "euler_74_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_74_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_81, "euler_74_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_75_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_82, "euler_75_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_75_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_82, "euler_75_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_76_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_83, "euler_76_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_76_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_83, "euler_76_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_77_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_84, "euler_77_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_77_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_84, "euler_77_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_77_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_85, "euler_77_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_77_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_85, "euler_77_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_77_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_86, "euler_77_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_77_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_86, "euler_77_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_77_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_87, "euler_77_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_77_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_87, "euler_77_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_78_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_88, "euler_78_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_78_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_88, "euler_78_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_78_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_89, "euler_78_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_78_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_89, "euler_78_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_78_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_90, "euler_78_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_78_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_90, "euler_78_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_78_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_91, "euler_78_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_78_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_91, "euler_78_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_79_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_92, "euler_79_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_79_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_92, "euler_79_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_79_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_93, "euler_79_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_79_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_93, "euler_79_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_79_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_94, "euler_79_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_79_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_94, "euler_79_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_79_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_95, "euler_79_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_79_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_95, "euler_79_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_80_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_96, "euler_80_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_80_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_96, "euler_80_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_80_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_97, "euler_80_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_80_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_97, "euler_80_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_80_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_98, "euler_80_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_80_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_98, "euler_80_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_80_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_99, "euler_80_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_80_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_99, "euler_80_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_81_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_100, "euler_81_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_81_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_100, "euler_81_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_81_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_101, "euler_81_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_81_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_101, "euler_81_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_82_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_102, "euler_82_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_82_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_102, "euler_82_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_82_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_103, "euler_82_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_82_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_103, "euler_82_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_82_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_104, "euler_82_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_82_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_104, "euler_82_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_82_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_105, "euler_82_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_82_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_105, "euler_82_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_83_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_106, "euler_83_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_83_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_106, "euler_83_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_84_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_107, "euler_84_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_84_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_107, "euler_84_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_85_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_108, "euler_85_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_85_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_108, "euler_85_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_86_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_109, "euler_86_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_86_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_109, "euler_86_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_87_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_110, "euler_87_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_87_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_110, "euler_87_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_88_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_111, "euler_88_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_88_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_111, "euler_88_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_89_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_112, "euler_89_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_89_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_112, "euler_89_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_90_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_113, "euler_90_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_90_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_113, "euler_90_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_91_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_114, "euler_91_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_91_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_114, "euler_91_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_92_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_115, "euler_92_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_92_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_115, "euler_92_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_93_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_116, "euler_93_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_93_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_116, "euler_93_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_94_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_117, "euler_94_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_94_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_117, "euler_94_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_95_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_118, "euler_95_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_95_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_118, "euler_95_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_96_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_119, "euler_96_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_96_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_119, "euler_96_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_97_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_120, "euler_97_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_97_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_120, "euler_97_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_98_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_121, "euler_98_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_98_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_121, "euler_98_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_98_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_122, "euler_98_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_98_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_122, "euler_98_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_99_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_123, "euler_99_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_99_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_123, "euler_99_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_99_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_124, "euler_99_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_99_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_124, "euler_99_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_100_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_125, "euler_100_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_100_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_125, "euler_100_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_100_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_126, "euler_100_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_100_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_126, "euler_100_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_101_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_127, "euler_101_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_101_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_127, "euler_101_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_101_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_128, "euler_101_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_101_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_128, "euler_101_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_102_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_129, "euler_102_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_102_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_129, "euler_102_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_102_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_130, "euler_102_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_102_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_130, "euler_102_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_103_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_131, "euler_103_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_103_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_131, "euler_103_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_104_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_132, "euler_104_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_104_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_132, "euler_104_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_105_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_133, "euler_105_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_105_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_133, "euler_105_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_106_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_134, "euler_106_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_106_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_134, "euler_106_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_106_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_135, "euler_106_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_106_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_135, "euler_106_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_107_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_136, "euler_107_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_107_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_136, "euler_107_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_108_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_137, "euler_108_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_108_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_137, "euler_108_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_109_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_138, "euler_109_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_109_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_138, "euler_109_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_110_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_139, "euler_110_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_110_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_139, "euler_110_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_111_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_140, "euler_111_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_111_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_140, "euler_111_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_112_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_141, "euler_112_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_112_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_141, "euler_112_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_113_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_142, "euler_113_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_113_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_142, "euler_113_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_114_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_143, "euler_114_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_114_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_143, "euler_114_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_115_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_144, "euler_115_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_115_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_144, "euler_115_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_115_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_145, "euler_115_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_115_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_145, "euler_115_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_115_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_146, "euler_115_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_115_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_146, "euler_115_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_115_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_147, "euler_115_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_115_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_147, "euler_115_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_116_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_148, "euler_116_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_116_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_148, "euler_116_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_116_0_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_149, "euler_116_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_116_0_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_149, "euler_116_0_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_116_1_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_150, "euler_116_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_116_1_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_150, "euler_116_1_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_116_1_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_151, "euler_116_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_116_1_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_151, "euler_116_1_1")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_117_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_152, "euler_117_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_117_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_152, "euler_117_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_118_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_153, "euler_118_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_118_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_153, "euler_118_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_119_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_154, "euler_119_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_119_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_154, "euler_119_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#euler_120_0_0(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_155, "euler_120_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_euler_120_0_0(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#output: &mut metal_oxide::Buffer<f32>, r#params: &metal_oxide::Buffer<f32>, r#args: r#EulerArgs, r#increments: &metal_oxide::Buffer<f32>, r#curve: &metal_oxide::Buffer<f32>, r#lift_decay: &metal_oxide::Buffer<f32>, r#lift_weight: &metal_oxide::Buffer<f32>, r#lift_drift_scale: &metal_oxide::Buffer<f32>, r#program: &metal_oxide::Buffer<f32>) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_155, "euler_120_0_0")?, config, &[metal_oxide::Argument::write(r#output), metal_oxide::Argument::read(r#params), metal_oxide::Argument::value(r#args)?, metal_oxide::Argument::read(r#increments), metal_oxide::Argument::read(r#curve), metal_oxide::Argument::read(r#lift_decay), metal_oxide::Argument::read(r#lift_weight), metal_oxide::Argument::read(r#lift_drift_scale), metal_oxide::Argument::read(r#program)])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#fft_columns_2(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_156, "fft_columns_2")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_fft_columns_2(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_156, "fft_columns_2")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#fft_columns_3(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_157, "fft_columns_3")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_fft_columns_3(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_157, "fft_columns_3")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#fft_columns_extract_1(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_158, "fft_columns_extract_1")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_fft_columns_extract_1(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_158, "fft_columns_extract_1")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#fft_columns_extract_2(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_159, "fft_columns_extract_2")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_fft_columns_extract_2(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_159, "fft_columns_extract_2")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#fft_columns_extract_3(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_160, "fft_columns_extract_3")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_fft_columns_extract_3(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#s0: u32, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#threads: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_160, "fft_columns_extract_3")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#s0)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::value(r#threads)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#sheet_generate(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#lam: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#cells: u32, r#cols: u32, r#seed: u32, r#first_cell: u64, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_161, "sheet_generate")?, config, &[metal_oxide::Argument::write(r#real), metal_oxide::Argument::write(r#imag), metal_oxide::Argument::read(r#lam), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#cols)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#first_cell)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_sheet_generate(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#lam: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#cells: u32, r#cols: u32, r#seed: u32, r#first_cell: u64, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_161, "sheet_generate")?, config, &[metal_oxide::Argument::write(r#real), metal_oxide::Argument::write(r#imag), metal_oxide::Argument::read(r#lam), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#cols)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#first_cell)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#sheet_transpose(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#real_t: &mut metal_oxide::Buffer<f32>, r#imag_t: &mut metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#cells: u32, r#rows: u32, r#cols: u32, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_162, "sheet_transpose")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::write(r#real_t), metal_oxide::Argument::write(r#imag_t), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#cols)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_sheet_transpose(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#imag: &metal_oxide::Buffer<f32>, r#real_t: &mut metal_oxide::Buffer<f32>, r#imag_t: &mut metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#cells: u32, r#rows: u32, r#cols: u32, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_162, "sheet_transpose")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::read(r#imag), metal_oxide::Argument::write(r#real_t), metal_oxide::Argument::write(r#imag_t), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#cols)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#sheet_extract(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#output: &mut metal_oxide::Buffer<f32>, r#cells: u32, r#rows: u32, r#m: u32, r#n: u32, r#r: f32, r#correction: f32, r#seed: u32, r#correction_cell: u64, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_163, "sheet_extract")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#m)?, metal_oxide::Argument::value(r#n)?, metal_oxide::Argument::value(r#r)?, metal_oxide::Argument::value(r#correction)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#correction_cell)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_sheet_extract(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &metal_oxide::Buffer<f32>, r#output: &mut metal_oxide::Buffer<f32>, r#cells: u32, r#rows: u32, r#m: u32, r#n: u32, r#r: f32, r#correction: f32, r#seed: u32, r#correction_cell: u64, r#total: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_163, "sheet_extract")?, config, &[metal_oxide::Argument::read(r#real), metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#cells)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#m)?, metal_oxide::Argument::value(r#n)?, metal_oxide::Argument::value(r#r)?, metal_oxide::Argument::value(r#correction)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#correction_cell)?, metal_oxide::Argument::value(r#total)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#sheet_fft_butterfly(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#n: u32, r#half_stride: u32, r#butterflies: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_164, "sheet_fft_butterfly")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::value(r#n)?, metal_oxide::Argument::value(r#half_stride)?, metal_oxide::Argument::value(r#butterflies)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_sheet_fft_butterfly(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#n: u32, r#half_stride: u32, r#butterflies: u32) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_164, "sheet_fft_butterfly")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::value(r#n)?, metal_oxide::Argument::value(r#half_stride)?, metal_oxide::Argument::value(r#butterflies)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#sheet_fft_tile(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#stages: u32, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_165, "sheet_fft_tile")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::value(r#stages)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_sheet_fft_tile(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#stages: u32, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_165, "sheet_fft_tile")?, config, &[metal_oxide::Argument::read_write(r#real), metal_oxide::Argument::read_write(r#imag), metal_oxide::Argument::value(r#stages)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#gen_tile_fft(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#eigs: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#tile_log: u32, r#seed: u32, r#base_cell: u64, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_166, "gen_tile_fft")?, config, &[metal_oxide::Argument::write(r#real), metal_oxide::Argument::write(r#imag), metal_oxide::Argument::read(r#eigs), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#tile_log)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#base_cell)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_gen_tile_fft(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#real: &mut metal_oxide::Buffer<f32>, r#imag: &mut metal_oxide::Buffer<f32>, r#eigs: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#tile_log: u32, r#seed: u32, r#base_cell: u64, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_166, "gen_tile_fft")?, config, &[metal_oxide::Argument::write(r#real), metal_oxide::Argument::write(r#imag), metal_oxide::Argument::read(r#eigs), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#tile_log)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#base_cell)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#gen_fft_extract(&self, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#eigs: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#seed: u32, r#base_cell: u64, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            self.device.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_167, "gen_fft_extract")?, config, &[metal_oxide::Argument::read(r#eigs), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#base_cell)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

    /// # Safety
    /// The caller must satisfy the kernel's bounds, race, and synchronization contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn r#enqueue_gen_fft_extract(&self, batch: &mut metal_oxide::Batch<'_>, config: impl std::convert::Into<metal_oxide::DynamicLaunchConfig>, r#eigs: &metal_oxide::Buffer<f32>, r#reverse: &metal_oxide::Buffer<u32>, r#twiddles: &metal_oxide::Buffer<f32>, r#log_n: u32, r#seed: u32, r#base_cell: u64, r#output: &mut metal_oxide::Buffer<f32>, r#size: u32, r#scale: f32, r#rows: u32, r#parity: u32, r#shared_real: usize, r#shared_imag: usize) -> metal_oxide::Result<()> {
        // SAFETY: kernel-specific preconditions are delegated to this function's caller.
        unsafe {
            batch.launch(self::metal_oxide_pipeline(self.device, &self.module, &self.pipeline_167, "gen_fft_extract")?, config, &[metal_oxide::Argument::read(r#eigs), metal_oxide::Argument::read(r#reverse), metal_oxide::Argument::read(r#twiddles), metal_oxide::Argument::value(r#log_n)?, metal_oxide::Argument::value(r#seed)?, metal_oxide::Argument::value(r#base_cell)?, metal_oxide::Argument::write(r#output), metal_oxide::Argument::value(r#size)?, metal_oxide::Argument::value(r#scale)?, metal_oxide::Argument::value(r#rows)?, metal_oxide::Argument::value(r#parity)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_real)?, metal_oxide::Argument::threadgroup::<f32>(r#shared_imag)?])
        }
    }

}
