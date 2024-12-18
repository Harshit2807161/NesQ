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
    
    gates = [30,40,50,60,70,80,90,100,110,120,130,140,150]
    
    columns = ['cd_N','cd_I', 'Cirq', 'Qiskit_basic', 'Qiskit_stochastic', 'Qiskit_sabre', 'PyTket', 'cd_M', 'cd_M_t','cd_G','cd_NO']
    df = pd.DataFrame(columns=columns, index=gates)
    
    time_columns = ['t_N', 't_Cirq', 't_Qiskit_basic', 't_Qiskit_stochastic', 't_Qiskit_sabre', 't_PyTket', 't_M', 't_M_t','t_G','t_NO']
    time_df = pd.DataFrame(columns=time_columns, index=gates)
    
    base_command = "python -m nesq --dataset random --hardware qx20 --search 200 --numitersG 200 --gates "
    start_time = time.time()
    
    for gate in gates:
        
        cd_N_list = []
        #cd_N_Q_list = []
        cd_I_list = []
        cirq_list = []
        qiskit_basic_list = []
        qiskit_stochastic_list = []
        qiskit_sabre_list = []
        pytket_list = []
        cd_M_list = []
        cd_M_t_list = []
        cd_G_list = []
        cd_NO_list = []
        
        t_N_list = []
        t_cirq_list = []
        t_qiskit_basic_list = []
        t_qiskit_stochastic_list = []
        t_qiskit_sabre_list = []
        t_pytket_list = []
        t_M_list = []
        t_M_t_list = []
        t_G_list = []
        t_NO_list = []
        
        for i in range(10):
            command = f"{base_command}{gate}"
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
                
                
                if t_N is not None: t_N_list.append(t_N)
                #if cd_N_Q is not None: cd_N_Q_list.append(cd_N_Q)
                if t_cirq is not None: t_cirq_list.append(t_cirq)
                if t_qiskit_dict:
                    t_qiskit_basic_list.append(t_qiskit_dict.get('t_basic'))
                    t_qiskit_stochastic_list.append(t_qiskit_dict.get('t_stochastic'))
                    t_qiskit_sabre_list.append(t_qiskit_dict.get('t_sabre'))
                if t_pytket is not None: t_pytket_list.append(t_pytket)
                if t_M is not None: t_M_list.append(t_M)
                if t_M_t is not None: t_M_t_list.append(t_M_t)
                if t_G is not None: t_G_list.append(t_G)
                if t_NO is not None: t_NO_list.append(t_NO)
                
                
                if cd_N is not None: cd_N_list.append(cd_N)
                if cd_I is not None: cd_I_list.append(cd_I)
                #if cd_N_Q is not None: cd_N_Q_list.append(cd_N_Q)
                if cirq is not None: cirq_list.append(cirq)
                
                if qiskit_dict.get('basic') is not None: qiskit_basic_list.append(qiskit_dict.get('basic'))
                if qiskit_dict.get('stochastic') is not None: qiskit_stochastic_list.append(qiskit_dict.get('stochastic'))
                if qiskit_dict.get('sabre') is not None: qiskit_sabre_list.append(qiskit_dict.get('sabre'))
                if pytket is not None: pytket_list.append(pytket)
                if cd_M is not None: cd_M_list.append(cd_M)
                if cd_M_t is not None: cd_M_t_list.append(cd_M_t)
                if cd_G is not None: cd_G_list.append(cd_G)
                if cd_NO is not None: cd_NO_list.append(cd_NO)
            
            
        curr_time = time.time()-start_time
        df.loc[gate] = [
            #np.mean(cd_N_Q_list) if cd_N_Q_list else np.nan,
            np.mean(cd_N_list) if cd_N_list else np.nan,
            np.mean(cd_I_list) if cd_I_list else np.nan,
            np.mean(cirq_list) if cirq_list else np.nan,
            np.mean(qiskit_basic_list) if qiskit_basic_list else np.nan,
            np.mean(qiskit_stochastic_list) if qiskit_stochastic_list else np.nan,
            np.mean(qiskit_sabre_list) if qiskit_sabre_list else np.nan,
            np.mean(pytket_list) if pytket_list else np.nan,
            np.mean(cd_M_list) if cd_M_list else np.nan,
            np.mean(cd_M_t_list) if cd_M_t_list else np.nan,
            np.mean(cd_G_list) if cd_G_list else np.nan,
            np.mean(cd_NO_list) if cd_NO_list else np.nan
            
        ]
        
        time_df.loc[gate] = [
            #np.mean(cd_N_Q_list) if cd_N_Q_list else np.nan,
            np.mean(t_N_list) if t_N_list else np.nan,
            np.mean(t_cirq_list) if t_cirq_list else np.nan,
            np.mean(t_qiskit_basic_list) if t_qiskit_basic_list else np.nan,
            np.mean(t_qiskit_stochastic_list) if t_qiskit_stochastic_list else np.nan,
            np.mean(t_qiskit_sabre_list) if t_qiskit_sabre_list else np.nan,
            np.mean(t_pytket_list) if t_pytket_list else np.nan,
            np.mean(t_M_list) if t_M_list else np.nan,
            np.mean(t_M_t_list) if t_M_t_list else np.nan,
            np.mean(t_G_list) if t_G_list else np.nan,
            np.mean(t_NO_list) if t_NO_list else np.nan
        ]

    df.to_csv('results_random_circuits_opt.csv', index_label='gates')
    time_df.to_csv('results_random_circuits_time_opt.csv', index_label='gates')
    