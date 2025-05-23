# msg

# Left arm position and command
float32[6] arx_pos_left  # 0-2: pos_x, pos_y, pos_z; 3-5: rot_x, rot_y, rot_z
int32 cmd_left # 1

# Right arm position and command
float32[6] arx_pos_right  # 0-2: pos_x, pos_y, pos_z; 3-5: rot_x, rot_y, rot_z
int32 cmd_right # 1

# end effects
float32[2] vcap #0-85
