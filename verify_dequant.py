import numpy as np
import json
import sys

def float16_to_float32(u16):
    return np.frombuffer(u16.to_bytes(2, 'little'), dtype=np.float16)[0].astype(np.float32)

def dequantize_q4_k(block_bytes):
    if len(block_bytes) != 144:
        raise ValueError(f"Block must be 144 bytes, got {len(block_bytes)}")

    d = float16_to_float32(int.from_bytes(block_bytes[0:2], 'little'))
    dmin = float16_to_float32(int.from_bytes(block_bytes[2:4], 'little'))
    
    scales = block_bytes[4:16]
    qs = block_bytes[16:144]
    
    sc = [0] * 8
    m = [0] * 8
    
    # get_scale_min_k4 logic
    for j in range(8):
        if j < 4:
            sc[j] = scales[j] & 63
            m[j] = scales[j+4] & 63
        else:
            sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
            m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
            
    D = [d * s for s in sc]
    M = [dmin * val for val in m]
    
    out = [0.0] * 256
    for j in range(8):
        for k in range(16):
            b = qs[j*16 + k]
            v0 = b & 0xF
            v1 = b >> 4
            
            # idx0 = j*32 + k
            # idx1 = j*32 + k + 16
            out[j*32 + k] = D[j] * v0 - M[j]
            out[j*32 + k + 16] = D[j] * v1 - M[j]
            
    return out, d, dmin, sc, m

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_dequant.py '<json_bytes>'")
        sys.exit(1)
        
    block_bytes = bytes(json.loads(sys.argv[1]))
    floats = dequantize_q4_k(block_bytes)
    print(json.dumps([float(f) for f in floats]))
