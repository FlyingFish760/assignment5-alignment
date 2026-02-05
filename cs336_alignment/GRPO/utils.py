# def clip(val: float, clip_range: float):
#     '''
#     clips the val within the range [1-clip_range, 1+clip_range]
#     '''
#     if val < 1 - clip_range: return 1 - clip_range
#     elif val > 1 + clip_range: return 1 - clip_range
#     else: return val