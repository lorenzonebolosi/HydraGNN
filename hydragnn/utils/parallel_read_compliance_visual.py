import os

import torch
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from torch_geometric.transforms import RadiusGraph, Distance
from torch_geometric.data import Data

#Used for splitting the inputs between all the processes
def split( a, n):
    k, m = divmod(len(a), n)
    final_list = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return list(final_list)

#########################################################################################
#Function that reads the i inputs , the i outputs  and the mesh and merges it all in a tensor
#########################################################################################

def convert_files_to_tensor(number_of_iterations, value):

    #Final list of all tensors
    final_list = []
    output_dir = 'compliance_graphs/'
   ################################
   #READ AND CONVERT U VALUES
   ################################
    compliance_array= [None] * 400
    for i in range(number_of_iterations):
        with open(os.getcwd()+"/freeFEM_results/"+ value +"/output_iteration_"+str(i)+".txt", 'r') as f:
          u_values = f.read().split('\n')

        #Need to remove the last row that is always empty
        end_index = len(u_values) - 1
        u_values = u_values[:end_index]
        u_values = [list(map(float, line.split())) for line in u_values]

        # Convert the list of lists into a PyTorch tensor
        u_tensor = torch.tensor(u_values, dtype=torch.float32)
        #torch.unique(u_tensor, dim=0)

        ################################
        # Read compliance
        ################################

        with open(os.getcwd() + "/compliance/" + value + "/compliance_" + str(i) + ".txt", 'r') as f:
            compliance = f.read().split('\n')

        # Need to remove the last row that is always empty
        end_index = len(compliance) - 1
        compliance = compliance[:end_index]
        compliance = [list(map(float, line.split())) for line in compliance]
        compliance = compliance[0] *4961


        # Convert the list of lists into a PyTorch tensor
        compliance_tensor = torch.tensor(compliance, dtype=torch.float32)
        compliance_tensor = torch.reshape(compliance_tensor, (4961, 1))
        compliance_array[i] = compliance_tensor.data[0]
        # torch.unique(compliance, dim=0)

        ################################
        #READ AND CONVERT W VALUES
        ################################

        with open(os.getcwd()+"/freeFEM_results/"+ value +"/input_iteration_"+str(i)+".txt", 'r') as f:
            w_values = f.read().split('\n')

        #Need to remove the last row that is always empty
        end_index = len(w_values) - 1
        w_values = w_values[:end_index]
        w_values = [list(map(float, line.split())) for line in w_values]

        # Convert the list of lists into a PyTorch tensor
        w_tensor = torch.tensor(w_values, dtype=torch.float32)
        #torch.unique(w_tensor, dim=0)
        output = torch.cat((mesh_tensor, u_tensor, w_tensor, compliance_tensor),1)
        final_list.append(output)

    plt.plot(compliance_array)
    plt.title('Compliance run:' + value)
    # Add x and y labels
    plt.xlabel('Iteration number')
    plt.ylabel('Compliance value')
    # Display the plot
    plt.tight_layout()

    filename = os.path.join(os.getcwd()+'/'+output_dir, value+'.png')
    plt.savefig(filename)
    plt.close()
    return final_list


#Mesh tensor only once
def convert_mesh_tensor():
   cwd = os.getcwd()
   print(cwd)
   with open(os.getcwd()+"/freeFEM_results/mesh.msh", 'r') as f:
      mesh = f.read().split('\n')

   #Split before and after the Vertices info
   end_index = int(mesh[0].split()[0]) + 1
   mesh = mesh[1:end_index]

   # Parse the string into numerical values
   #data_lines = mesh.strip().split('\n')
   mesh_values = [list(map(float, line.split())) for line in mesh]

   # Convert the list of lists into a PyTorch tensor
   mesh_tensor = torch.tensor(mesh_values, dtype=torch.float32)
   mesh_tensor = mesh_tensor[: , [0,1]]
   #print(mesh_tensor.size())
   # Print the resulting tensor
   #print(mesh_tensor)
   return mesh_tensor

mesh_tensor = convert_mesh_tensor()

def read_freeFEM_results(value):
    # Dictionary of all the tensors
    tensordictionary = []
    # Get the length of the result folder
    _, _, files = next(os.walk(os.getcwd() + "/freeFEM_results/" + value + "/"))
    files.sort()
    # -1 for the mesh and -1 for .DS_store. /2 for input output
    file_count = int((len(files) - 2) / 2)
    tensordictionary = tensordictionary + convert_files_to_tensor(file_count, value)
    print("Succesfully converted: " + str(len(tensordictionary)) + " iterations data")
    print("Each tensor has size: " + str(tensordictionary[0].size()))
    return tensordictionary

def parallel_processing(data):

    tensordictionary = read_freeFEM_results(data)
    graph_list = []
    create_graph_fromXYZ = RadiusGraph(r=5.0)
    compute_edge_lengths = Distance(norm=False, cat=True)

    #Create the edge only once
    first_data = Data()
    first_data.pos = tensordictionary[0][:, :2]
    first_data.x = tensordictionary[0][:, 2:]
    first_data = create_graph_fromXYZ(first_data)
    first_data = compute_edge_lengths(first_data)

    #Convert tensors in Data objects
    for tensor in tensordictionary:

        #print(str(type(tensor)))
        # I need to pass a data that has the discussed structure
        data = Data()
        data.pos = tensor[:, :2]
        data.x = tensor[:, 2:5]
        data.y = tensor[:, 5:][0]
        data.edge_index = first_data.edge_index
        data.edge_attr = first_data.edge_attr
        graph_list.append(data)
    return graph_list
