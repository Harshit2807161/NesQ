
import subprocess
import shlex
import pandas as pd
import numpy as np
import re
import ast
import time
import threading
import queue



def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def run_script_with_args(args):
    process = subprocess.Popen(shlex.split(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    q_stdout = queue.Queue()
    q_stderr = queue.Queue()
    
    stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, q_stdout))
    stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, q_stderr))
    
    stdout_thread.start()
    stderr_thread.start()
    
    output = []
    
    while True:
        try:
            line = q_stdout.get_nowait()
        except queue.Empty:
            pass
        else:
            print(line, end='')  # Print to console
            output.append(line)
        
        try:
            line = q_stderr.get_nowait()
        except queue.Empty:
            pass
        else:
            print(line, end='')  # Print to console
            output.append(line)
        
        if process.poll() is not None:
            break
    
    stdout_thread.join()
    stderr_thread.join()
    
    returncode = process.returncode
    if returncode == 0:
        return ''.join(output)
    else:
        print(f"Command failed with return code {returncode}")
        return None



def parse_output_for_value(output, key):
    for line in output.split('\n'):
        if key in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                print("Oh no")
                return None
    return None

def parse_output_for_dict(output, key):
    for line in output.split('\n'):
        if key in line:
            try:
                dict_str = re.search(r"\{.*\}", line).group(0)
                return ast.literal_eval(dict_str)
            except (ValueError, AttributeError):
                print("Oh no dict")
                return None
    return None


if __name__ == "__main__":
    small_files = [
    "3_17_13","4gt10-v1_81","4gt11_82","4gt11_83",
    "4gt11_84","4gt12-v0_88","4gt12-v1_89","4gt13-v1_93","4gt13_90","4gt13_91",
    "4gt13_92","4gt4-v0_80",
    "4gt5_75","4gt5_76","4gt5_77",
    "4mod5-bdd_287","4mod5-v0_18","4mod5-v0_19",
    "4mod5-v0_20","4mod5-v1_22",
    "4mod5-v1_23",'4mod5-v1_24',
    '4mod7-v0_94',"4mod7-v1_96",
    "4_49_16","aj-e11_165",
    "alu-bdd_288",'alu-v0_26',
    'alu-v0_27_example','alu-v0_27',
    'alu-v1_28',"alu-v1_29",
    'alu-v2_32','alu-v2_33',
    'alu-v3_34','alu-v3_35',
    'alu-v4_36','alu-v4_37',
    'cnt3-5_179','decod24-bdd_294',
    'decod24-v0_38','decod24-v1_41',
    'decod24-v2_43','decod24-v3_45',
    'ex-1_166','ex1_226',
    'graycode6_47','ham3_102',
    'ising_model_10','miller_11',
    'mini_alu_305','mod10_176',
    'mod5d1_63','mod5d2_64',
    'mod5mils_65','one-two-three-v0_98',
    'one-two-three-v1_99','one-two-three-v2_100',
    'one-two-three-v3_101','qft_10',
    'rd32-v0_66','rd32-v1_68',
    'rd32_270','rd53_138',
    'sys6-v0_111','xor5_254']      

    columns = ['cd_N','cd_I', 'Cirq', 'Qiskit_basic', 'Qiskit_stochastic', 'Qiskit_sabre', 'PyTket', 'cd_M', 'cd_M_t','cd_G','cd_NO']
    df = pd.DataFrame(columns=columns, index=small_files)
    
    time_columns = ['t_N', 't_Cirq', 't_Qiskit_basic', 't_Qiskit_stochastic', 't_Qiskit_sabre', 't_PyTket', 't_M', 't_M_t','t_G','t_NO']
    time_df = pd.DataFrame(columns=time_columns, index=small_files)
    
    base_command = "python -m nesq --dataset small --hardware qx20 --search 200 --numitersG 200 --small_file "
    start_time = time.time()
    

    for small_file in small_files:
        
        cd_N_list = [1000000000000]
        #cd_N_Q_list = []
        cd_I_list = [1000000000000]
        cirq_list = [1000000000000]
        qiskit_basic_list = [1000000000000]
        qiskit_stochastic_list = [1000000000000]
        qiskit_sabre_list = [1000000000000]
        pytket_list = [1000000000000]
        cd_M_list = [1000000000000]
        cd_M_t_list = [1000000000000]
        cd_G_list = [1000000000000]
        cd_NO_list = [1000000000000]
        
        t_N_list = [1000000000000]
        t_cirq_list = [1000000000000]
        t_qiskit_basic_list = [1000000000000]
        t_qiskit_stochastic_list = [1000000000000]
        t_qiskit_sabre_list = [1000000000000]
        t_pytket_list = [1000000000000]
        t_M_list = [1000000000000]
        t_M_t_list = [1000000000000]
        t_G_list = [1000000000000]
        t_NO_list = [1000000000000]
        
        
        command = f"{base_command}{small_file}"
        output = run_script_with_args(command)
        if output:
            cd_N = parse_output_for_value(output, "Output circuit depth for N")
            #cd_N_Q = parse_output_for_value(output, "Output circuit depth for N with Qiskit")
            cd_I = parse_output_for_value(output, "Layers in input circuit")
            cirq = parse_output_for_value(output, "Cirq Routing Distance")
            qiskit_dict = parse_output_for_dict(output, "Qiskit Routing Distance")
            pytket = parse_output_for_value(output, "PyTket Routing Distance")
            cd_M = parse_output_for_value(output, "Output circuit depth for M")
            cd_M_t = parse_output_for_value(output, "Output circuit depth for M with Training")
            cd_G = parse_output_for_value(output, "Output circuit depth for G")
            cd_NO = parse_output_for_value(output, "Output circuit depth for NO")
            
            t_N = parse_output_for_value(output, "Time N")
            #cd_N_Q = parse_output_for_value(output, "Output circuit depth for N with Qiskit")
            t_cirq = parse_output_for_value(output, "Time Cirq")
            t_qiskit_dict = parse_output_for_dict(output, "Time Qiskit")
            t_pytket = parse_output_for_value(output, "Time PyTket")
            t_M = parse_output_for_value(output, "Time M")
            t_M_t = parse_output_for_value(output, "Time M with Training")
            t_G = parse_output_for_value(output, "Time G")
            t_NO = parse_output_for_value(output, "Time NO")
            
            
            
            if t_N is not None: t_N_list[0] = t_N
            #if cd_N_Q is not None: cd_N_Q_list.append(cd_N_Q)
            if t_cirq is not None: t_cirq_list[0] = t_cirq
            if t_qiskit_dict:
                t_qiskit_basic_list[0] = t_qiskit_dict.get('t_basic')
                t_qiskit_stochastic_list[0] = t_qiskit_dict.get('t_stochastic')
                t_qiskit_sabre_list[0] = t_qiskit_dict.get('t_sabre')
            if t_pytket is not None: t_pytket_list[0] = t_pytket
            if t_M is not None: t_M_list[0] = t_M
            if t_M_t is not None: t_M_t_list[0] = t_M_t
            if t_G is not None: t_G_list[0] = t_G
            if t_NO is not None: t_NO_list[0] = t_NO
            
            if cd_N is not None: cd_N_list[0] = cd_N
            if cd_I is not None: cd_I_list[0] = cd_I
            #if cd_N_Q is not None: cd_N_Q_list.append(cd_N_Q)
            if cirq is not None: cirq_list[0] = cirq
            
            if qiskit_dict.get('basic') is not None: qiskit_basic_list[0] = qiskit_dict.get('basic')
            if qiskit_dict.get('stochastic') is not None: qiskit_stochastic_list[0] = qiskit_dict.get('stochastic')
            if qiskit_dict.get('sabre') is not None: qiskit_sabre_list[0] = qiskit_dict.get('sabre')
            if pytket is not None: pytket_list[0] = pytket
            if cd_M is not None: cd_M_list[0] = cd_M
            if cd_M_t is not None: cd_M_t_list[0] = cd_M_t
            if cd_G is not None: cd_G_list[0] = cd_G
            if cd_NO is not None: cd_NO_list[0] = cd_NO
            
            
        curr_time = time.time()-start_time
        df.loc[small_file] = [
            #np.mean(cd_N_Q_list) if cd_N_Q_list else np.nan,
            cd_N_list[0] if cd_N_list else np.nan,
            cd_I_list[0] if cd_I_list else np.nan,
            cirq_list[0] if cirq_list else np.nan,
            qiskit_basic_list[0] if qiskit_basic_list else np.nan,
            qiskit_stochastic_list[0] if qiskit_stochastic_list else np.nan,
            qiskit_sabre_list[0] if qiskit_sabre_list else np.nan,
            pytket_list[0] if pytket_list else np.nan,
            cd_M_list[0] if cd_M_list else np.nan,
            cd_M_t_list[0] if cd_M_t_list else np.nan,
            cd_G_list[0] if cd_G_list else np.nan,
            cd_NO_list[0] if cd_NO_list else np.nan
        ]
        
        time_df.loc[small_file] = [
            #np.mean(cd_N_Q_list) if cd_N_Q_list else np.nan,
            t_N_list[0] if t_N_list else np.nan,
            t_cirq_list[0] if t_cirq_list else np.nan,
            t_qiskit_basic_list[0] if t_qiskit_basic_list else np.nan,
            t_qiskit_stochastic_list[0] if t_qiskit_stochastic_list else np.nan,
            t_qiskit_sabre_list[0] if t_qiskit_sabre_list else np.nan,
            t_pytket_list[0] if t_pytket_list else np.nan,
            t_M_list[0] if t_M_list else np.nan,
            t_M_t_list[0] if t_M_t_list else np.nan,
            t_G_list[0] if t_G_list else np.nan,
            t_NO_list[0] if t_NO_list else np.nan
            
        ]

    df.to_csv('results_small_circuits_opt.csv', index_label='circuit name')
    time_df.to_csv('results_time_small_circuits_opt.csv', index_label='circuit name')
    