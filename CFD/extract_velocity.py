import data_functions
import yaml
import os

def main():
    def read_data_extraction_inputs():
        print(os.getcwd())
        with open(f'../{data_functions.data_extraction_inputs}', 'r') as f:
            lines = f.readlines()
            path = lines[0].strip()
            t_start, t_end, t_step = lines[1].strip(), lines[2].strip(), lines[3].strip()

        return [path, int(t_start), int(t_end), int(t_step)]


    def output_sensor_data_as_yaml(path, data):
        with open(f'{path}/{data_functions.sensor_data_yaml}', 'w') as f:
            yaml.dump(data, f)

    inputs = read_data_extraction_inputs()
    data = data_functions.mp_extract_velocity(*inputs)
    output_sensor_data_as_yaml(inputs[0], data)

if __name__ == '__main__':
    main()


