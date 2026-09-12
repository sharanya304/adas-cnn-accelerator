import torch
import numpy as np

# Fixed-point configuration
TOTAL_BITS = 16  # Bit-width (e.g., 16-bit signed integer)
FRAC_BITS = 8    # Q8.8 format (8 fractional bits)

def float_to_fixed(tensor, total_bits=TOTAL_BITS, frac_bits=FRAC_BITS):
    """Converts a PyTorch tensor to a signed fixed-point integer array."""
    tensor = tensor.to(torch.float32)
    scaled = torch.round(tensor * (2 ** frac_bits))
    
    min_val = -(1 << (total_bits - 1))
    max_val = (1 << (total_bits - 1)) - 1
    clipped = torch.clamp(scaled, min=min_val, max=max_val)
    
    return clipped.cpu().detach().numpy().astype(np.int32)

def export_readable_text_weights(weights_path, output_txt="fixed_point_weights.txt"):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    
    # Unwrap state_dict if checkpoint contains nested metadata dictionaries
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    stages = ['patch_stage', 'layer_stage', 'head_trunk', 'cls_head', 'wh_head']
    
    # Group parameters by stage
    grouped_data = {stage: {} for stage in stages}
    grouped_data['other'] = {}

    for key, tensor in state_dict.items():
        clean_key = key.replace('module.', '')
        parts = clean_key.split('.')
        stage_name = parts[0]
        param_name = '.'.join(parts[1:]) if len(parts) > 1 else 'value'
        
        fixed_array = float_to_fixed(tensor)
        
        if stage_name in grouped_data:
            grouped_data[stage_name][param_name] = fixed_array
        else:
            grouped_data['other'][clean_key] = fixed_array

    # Write formatted output to text file
    with open(output_txt, 'w') as f:
        f.write("========================================================================\n")
        f.write(f" MCUNetV2 QUANTIZED WEIGHTS & PARAMETERS (Q{TOTAL_BITS - FRAC_BITS}.{FRAC_BITS} Format)\n")
        f.write(f" Total Bits: {TOTAL_BITS} | Fractional Bits: {FRAC_BITS}\n")
        f.write("========================================================================\n\n")

        for stage, params in grouped_data.items():
            if not params:
                continue
            
            f.write("########################################################################\n")
            f.write(f" STAGE: {stage.upper()}\n")
            f.write("########################################################################\n\n")

            for param_name, arr in params.items():
                f.write(f"------------------------------------------------------------------------\n")
                f.write(f" Parameter: {stage}.{param_name}\n")
                f.write(f" Shape:     {list(arr.shape)}\n")
                f.write(f" Data Type: int32 (Q{TOTAL_BITS - FRAC_BITS}.{FRAC_BITS})\n")
                f.write(f"------------------------------------------------------------------------\n")

                # Format 1D vectors (biases, batchnorm params) clean on single lines/compact blocks
                if arr.ndim <= 1:
                    f.write(np.array2string(arr, max_line_width=100, threshold=100000) + "\n\n")
                
                # Format 2D/3D/4D tensors (conv/linear weights) with indented blocks
                else:
                    # Print full array without truncation
                    formatted_str = np.array2string(arr, max_line_width=120, threshold=100000)
                    f.write(formatted_str + "\n\n")

    print(f"Successfully generated human-readable weight file: '{output_txt}'")

if __name__ == "__main__":
    CHECKPOINT_PATH = "mcunetv2_40epochs.pth"
    export_readable_text_weights(CHECKPOINT_PATH)