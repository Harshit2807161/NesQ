# NesQ: Nested Monte Carlo Search-Based Qubit Router

NesQ is an innovative qubit router utilizing the Nested Monte Carlo Search (NMCS) technique to optimize qubit routing, thereby improving the execution of quantum circuits. This tool is especially crucial in the NISQ (Noisy Intermediate-Scale Quantum) era, where limited qubit connectivity on quantum processing units poses significant challenges to efficient quantum task scheduling.

Current submission link: https://drive.google.com/file/d/1IbsQ_zFnPByd5-KSsRr3L9CRraxdQpHR/view?usp=drive_link

## Abstract

In the current NISQ era, it has become difficult to schedule increasingly complex quantum tasks with limited connectivity of the QPU (Quantum Processing Unit). This challenge arises from the requirement for qubits sharing a task to be physically connected in the hardware topology. Connectivity constraints necessitate the use of SWAP gates or reversing existing CNOT gates, which adds computational costs and introduces errors. Efficient routing agents are thus vital to optimizing qubit routing. We present NesQ, a Nested Monte Carlo Search (NMCS) based qubit router designed to efficiently explore the state space and optimize this routing problem. Our experiments demonstrate that NesQ outperforms existing routing algorithms, providing superior performance with reduced runtime.

## Environment Setup

To work with NesQ, you need to set up a suitable Python environment. Follow these steps:

1. **Create a Virtual Conda Environment**  
   Ensure you have Conda installed, then create a new environment named `nesqubit` using the requirements file provided in the repository:

   ```sh
   conda create --name nesqubit --file requirements.txt
   ```

2. **Set Current Working Directory**  
   Move your terminal or command prompt to the `code_data_and_results` directory within the repository. This is essential for running the routing experiments:

   ```sh
   cd code_data_and_results
   ```

## Running NesQ for Random and Large-Scale Circuits

NesQ can handle various types of circuits. Utilize the following command line prompts for different scenarios:

- **Large Circuits (`large`)**
  ```sh
  python -m nesq --dataset large --hardware qx20 --search 250 --numitersG 250 --large_files {name_of_large_circuit}
  ```

- **Small Circuits (`small`)**
  ```sh
  python -m nesq --dataset small --hardware qx20 --search 200 --numitersG 200 --small_file  {name_of_small_circuit}
  ```

- **Random Circuits (`random`)**
  ```sh
  python -m nesq --dataset random --hardware qx20 --search 200 --numitersG 200 --gates
  ```

**Note:** The parameters `search` and `numitersG` define the search depth for implementing the Qroute agent and the iterations in GNRPAm, respectively. You may adjust these according to your needs, but the values noted above were used in our experiments.

## Reproducing Results from Our AAAI 2025 Paper

To reproduce the results featured in our AAAI (QC+AI) 2025 paper, follow these steps:

1. **Create Virtual Conda Environment**

   Follow the command below if you haven't set up the environment yet:

   ```sh
   conda create --name nesqubit --file requirements.txt
   ```

2. **Set Current Working Directory**

   Navigate to the `code_data_and_results` directory:

   ```sh
   cd code_data_and_results
   ```

3. **Run Experiments**

   The experiments are saved as Python scripts with preset arguments. Execute them as per your requirement:

   - **Scalability on Random Circuits**: Run `random_circ_benchmark.py`
   - **Realistic Circuit Benchmarks (Small)**: Run `small_circ_benchmark.py`
   - **Realistic Circuit Benchmarks (Large)**: Run `large_circ_benchmark.py`
   - **Generalizability Across Device Topologies**: Run `device_large_benchmark.py`

---

Should you have any questions or require further assistance, feel free to create an issue in the repository. We look forward to your valuable feedback and contributions!

