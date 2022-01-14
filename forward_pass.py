from math import e as e
# define given inputs
b = 0.5
x = [1, 4, 5]
wh1 = [0.1, 0.3, 0.5]
wh2 = [0.2, 0.4, 0.6]
wo1 = [0.7, 0.9]
wo2 = [0.8, 0.1]
wh = [wh1, wh2]
wo = [wo1, wo2]
w = [wh, wo]
# ---------------------------------------------


# calculate forward pass
def get_z_output(input, w_out, bias, rnge):
    in_out = 0
    for i in range(rnge):
        # print(f'input[i]:{input[i]} , w_out[i]:{w_out[i]}')
        in_out = in_out + (input[i] * w_out[i])
    return round(in_out + bias, 4)


def calc_sigmoid(x_input):
    return 1 / (1 + e ** (-x_input))


def calculate_in_out(input_list, w_list):
    output_list = []
    for w in w_list:
        z_out = get_z_output(input_list, w, b, len(w))
        # print(f'zh1:{zh1}')
        out = round(calc_sigmoid(z_out), 4)
        output_list.append(out)
        # print(f'h1:{h1}')
    return output_list


def forward_pass(all_w_list):
    for w_list in all_w_list:
        output_list = calculate_in_out(x, w_list)
        print(f'output_list: {output_list}')


# ---------------------------------------------
