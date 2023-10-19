pub struct AttnEngImpl {
    // Model related parameters
    head_dim: u32,
    num_heads: u32,
    seq_len: u32,
    /// the actual 'window' size is 2 x window_size
    window_size: u32,

    // Hardware related parameters
    /// if the floating point MAC is fully pipelined then the latency is 1 cycle
    /// otherwise it is defined by "fmul_latency"
    FMAC_pipeline_II: u32,
    FACC_pipeline_II: u32,
    fmul_latency: u32,
    exp_latency: u32,
    frequency: u32,
    is_load_store_buffered: bool,
    num_attn_unit: u32,
}

impl AttnEngImpl {
    pub fn attention(&self) -> u32 {
        let num_cycles = attn_row() * seq_len;
        return num_cycles * self.frequency;
    }

    pub fn row_attention(&self) -> u32 {
        let time: u32 = 0;
        if (!is_load_store_buffered) {
            time += load_qkv();
            time += store_back();
        }

        // QK are performed in parallel
        // the number of repeating all engines depends on the size of the window
        let full_window_size = 2 * window_size;
        let num_repeating_per_row = if (full_window_size % num_attn_unit == 0) {
            full_window_size / num_attn_unit
        } else {
            full_window_size / num_attn_unit + 1
        };

        // count the time to process one row (exponential time is counted)
        let one_iter_time = qk_dot() + sv_mul() + z_reduction();
        time += num_repeating_per_row * one_iter_time;
    }

    fn load_qkv(&self) -> u32 {
        return self.head_dim;
    }

    fn qk_dot(&self) -> u32 {
        let qk_dot_latency = if (is_fmacc_pipeline) {
            self.fmul_latency + self.head_dim - 1;
        } else {
            self.fmul_latency + FMAC_pipeline_II * self.head_dim;
        };

        return qk_dot_latency + exp_latency;
    }

    fn sv_mul(&self) -> u32 {
        self.fmul_latency + self.head_dim - 1;
    }

    fn z_reduction(&self) -> u32 {
        self.FACC_pipeline_II * self.head_dim;
    }

    fn store_back(&self) -> u32 {
        self.head_dim;
    }
}
