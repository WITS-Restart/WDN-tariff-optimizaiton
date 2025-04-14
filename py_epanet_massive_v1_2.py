import sys
import wntr
import csv
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime
import os

#!VD!3#
import collections
lookup_table = collections.OrderedDict()
MAX_LOOKUP_SIZE = 13000 #per evitare che va fuori memoria
#!VD!3#
def get_lookup_table():
    return lookup_table


list_of_fixed_nodes_with_leaks = None

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

def pick_uniform_leaks(wn, number_of_junctions_with_leaks):
    print("**************** TEST leak uniform : ", number_of_junctions_with_leaks)

    node_names = wn.junction_name_list
    selected_junctions = []
    node_count = 0
    for ii in range(0,len(node_names),10):
        selected_junctions.append(node_names[ii])
        node_count += 1
        if node_count>number_of_junctions_with_leaks:
            break

    return selected_junctions

def pick_rand_leaks(wn, number_of_junctions_with_leaks):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, number_of_junctions_with_leaks)

    return selected_junctions

def pick_rand_group_leaks(wn, number_of_junctions_with_leaks, leak_group, leak_node):
    node_names = ["8614", "8600", "8610", "9402", "8598", "8608", "8620", "8616", "4922", "J106", "8618", "8604", "8596", "9410", "8612", "8602", "8606", "5656", "8622",
                      "8624", "8626", "8628", "8630", "8644", "8634", "8632", "8636", "8646", "8688", "8640", "8642", "8638", "8698", "8692", "8648", "8690", "8718",
                      "8702", "8700", "8694", "8738", "8696", "8740", "8720", "8706", "8704", "8686", "8708", "8660", "8656", "8664", "8662", "8654", "8716", "8650",
                      "8746", "8732", "8684", "8668", "8730", "8658", "8678", "8652", "8676", "8714", "8710", "8712", "8682", "8666", "8674", "8742", "8680", "8672",
                      "8792", "8722", "8726", "8724", "8744", "8736", "8728", "8670", "8734", "7384"]

    # random_group = np.random.random_integers(8)
    print("**************** TEST leak group : ", leak_group)
    random_group = leak_group #5
    selected_junctions = node_names[((random_group-1)*10)+leak_node:((random_group-1)*10)+leak_node+1]

    return selected_junctions




def assign_leaks(wn, area_size, selected_junctions):
    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

def create_custom_pattern(wn, name, min_mult, max_mult, step, duration):
    timeops = wntr.network.options.TimeOptions(duration)

    multipliers = []

    for multiplier_step in range(min_mult,max_mult+step,step):
        multipliers.append(multiplier_step)

    out_pattern = wntr.network.Pattern("custom", multipliers, time_options=timeops)

    wn.add_pattern(name, out_pattern)

    return out_pattern

def assign_rand_demand_to_junctions(wn, min_bd, max_bd, pattern=None):
    node_names = wn.junction_name_list
    # print("NUM NODE : ")
    # print(len(node_names))

    for juncID in node_names:

        # print(juncID)

        junc_obj = wn.get_node(juncID)

        new_demand = random.uniform(min_bd, max_bd)

        # junc_obj.demand_timeseries_list[0].base_value = new_demand

        # you can't apparently just change the old pattern like the base demand, it seems to be a read only field.
        # for now we just create a new timeseries and delete the old one
        # junc_obj.demand_timeseries_list[0].pattern = pattern

        junc_obj.add_demand(base=new_demand, pattern_name=pattern)
        del junc_obj.demand_timeseries_list[0]

        # To keep track of the history of the different random base demands we add a custom field to the Junction object of wntr
        # if it is the first time that we call this, we have to create it, else we just append the new value to the list
        if hasattr(junc_obj, 'list_of_bds'):
            junc_obj.list_of_bds.append(junc_obj.base_demand)
        else:
            junc_obj.list_of_bds = [junc_obj.base_demand]

def write_results_to_csv(results, results_link, sim_duration, wn, out_filename, number_of_nodes_with_leaks, fixed_demand, pattern_len, file_timestamp=False):
    print("Printing Nodes CSV. Please wait...")

    link_names = wn.link_name_list

    node_names = wn.node_name_list
    # node_names = ["8614", "8600", "8610", "9402", "8598", "8608", "8620", "8616", "4922", "J106", "8618", "8604", "8596", "9410", "8612", "8602", "8606", "5656", "8622",
    #                   "8624", "8626", "8628", "8630", "8644", "8634", "8632", "8636", "8646", "8688", "8640", "8642", "8638", "8698", "8692", "8648", "8690", "8718",
    #                   "8702", "8700", "8694", "8738", "8696", "8740", "8720", "8706", "8704", "8686", "8708", "8660", "8656", "8664", "8662", "8654", "8716", "8650",
    #                   "8746", "8732", "8684", "8668", "8730", "8658", "8678", "8652", "8676", "8714", "8710", "8712", "8682", "8666", "8674", "8742", "8680", "8672",
    #                   "8792", "8722", "8726", "8724", "8744", "8736", "8728", "8670", "8734", "7384"]

    nodes_index = [*range(len(node_names))]

    # print(results.node)
    # sys.exit(1)


    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']
    leak_demand_results = results.node["leak_demand"]

    flowrate_results = results_link.node['flowrate']
    velocity_results = results_link.node['velocity']
    status_results = results_link.node['status']

    sim_duration_in_hours = int(sim_duration / 3600)

    now = formatted_datetime()

    if (file_timestamp):
        out_filename_complete = out_filename + "_nodes_output_" + now + ".csv"
        out_filename_complete_link = out_filename + "_links_output_" + now + ".csv"
    else:
        out_filename_complete = out_filename + "_nodes_output.csv"
        out_filename_complete_link = out_filename + "_links_output.csv"

    #out = open(out_filename_complete, "w", newline='', encoding='utf-8')
    #writer = csv.writer(out, delimiter=';')

    header = ["hour", "nodeID", "cluster_demand", "base_demand", "demand_value", "head_value",
              "pressure_value", "x_pos", "y_pos", "node_type", "has_leak",
              "leak_area_value", "leak_discharge_value",
              "leak_demand_value",
              # "tot_junctions_demand", "tot_leaks_demand", "tot_network_demand",
              #"end_node_link",
              # "lost_0", "head_0", "pressure_0", "lost_1", "head_1", "pressure_1", "lost_2", "head_2", "pressure_2",
              # "lost_3", "head_3", "pressure_3", "lost_4", "head_4", "pressure_4", "lost_5", "head_5", "pressure_5",
              # "lost_6", "head_6", "pressure_6", "lost_7", "head_7", "pressure_7", "lost_8", "head_8", "pressure_8",
              # "lost_9", "head_9", "pressure_9"
              # "flow_demand_in",
              # "demand_0", "head_0", "pressure_0", "demand_1", "head_1", "pressure_1", "demand_2", "head_2", "pressure_2",
              # "demand_3", "head_3", "pressure_3", "demand_4", "head_4", "pressure_4", "demand_5", "head_5", "pressure_5",
              # "demand_6", "head_6", "pressure_6", "demand_7", "head_7", "pressure_7", "demand_8", "head_8", "pressure_8",
              # "demand_9", "head_9", "pressure_9",
              # "flow_demand_out",
              # "leak_group",
              ]

    #writer.writerow(header)
    df = pd.DataFrame(columns=header)

    #out_link = open(out_filename_complete_link, "w", newline='', encoding='utf-8')
    #writer_link = csv.writer(out_link, delimiter=';')

    #header_link = ["hour", "linkID", "link_type", "start_node", "end_node", "flowrate", "velocity", "status"]

    #writer_link.writerow(header_link)


    # These two variables are needed for the simulation stats
    tot_juncts_demand_in_entire_simulation = np.float64(0)
    tot_juncts_leak_demand_in_entire_simulation = np.float64(0)

    nodeIdentification = {}
    nodeIndex = 880000
    for nodeID in node_names:
        flag = True
        try:
            # try converting to integer
            int(nodeID)
        except ValueError:
            flag = False

        if flag:
            nodeIdentification[nodeID] = int(nodeID)
        else:
            nodeIdentification[nodeID] = nodeIndex
        nodeIndex += 1


    linkIdentification = {}
    linkIndex = 550000
    for linkID in link_names:
        flag = True
        try:
            # try converting to integer
            int(linkID)
        except ValueError:
            flag = False

        if flag:
            linkIdentification[linkID] = int(linkID)
        else:
            linkIdentification[linkID] = linkIndex
        linkIndex += 1


    for timestamp in range(sim_duration_in_hours):

        # Water network balancing counters reset each hour
        tot_leaks_demand = np.float64(0)
        tot_junctions_demand = np.float64(0)
        tot_network_demand = np.float64(0)

        nodeGroupInfo = []
        nodeIndex = 0
        pipeIndex = 0


        hour_in_seconds = int(timestamp * 3600)
        # flow_demand = demand_results.loc[hour_in_seconds, "7384"]*-1

        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)
            node_type = node_obj.__class__.__name__

            # if nodeIndex%10==0:
            #     # print(nodeIndex)
            #     group_has_leak = False
            #     nodeGroupInfo = []
            #     nodeGroupInfo.append(flow_demand)
            #
            #     for nodeIndexGroup in nodes_index[nodeIndex:nodeIndex+10]:
            #         node_group_obj = wn.get_node(node_names[nodeIndexGroup])
            #         node_group_type = node_group_obj.__class__.__name__
            #         leak_area_value = node_group_obj.leak_area
            #         if (leak_area_value > 0.0):
            #             group_has_leak = True  # this leak-flag is set to true if we see a hole in the node
            #
            #         if node_group_type == "Junction":
            #             if hasattr(node_group_obj, 'list_of_bds'):
            #                 # happens if we set random_base_demands to True
            #                 base_demand_group = node_group_obj.list_of_bds[timestamp]
            #             else:
            #                 base_demand_group = node_group_obj.demand_timeseries_list[0].base_value
            #         else:
            #             base_demand_group = 0.0
            #
            #         demand_value_group = demand_results.loc[hour_in_seconds, node_names[nodeIndexGroup]]
            #
            #         flow_demand = flow_demand - demand_value_group
            #
            #         demand_lost_group = base_demand_group - demand_value_group
            #         demand_lost_group = "{:.8f}".format(demand_lost_group)
            #         base_demand_group = "{:.8f}".format(base_demand_group)
            #
            #         demand_value_group = "{:.8f}".format(demand_value_group)
            #
            #         head_value_group = head_results.loc[hour_in_seconds, node_names[nodeIndexGroup]]
            #         head_value_group = "{:.8f}".format(head_value_group)
            #
            #         pressure_value_group = pressure_results.loc[hour_in_seconds, node_names[nodeIndexGroup]]
            #         pressure_value_group = "{:.8f}".format(pressure_value_group)
            #
            #         # print(nodeID)
            #         # print(base_demand_group)
            #         # print(demand_value_group)
            #         # print(demand_lost_group)
            #         # print(head_value_group)
            #         # print(pressure_value_group)
            #         # # sys.exit(1)
            #         # nodeGroupInfo.extend((demand_lost_group,head_value_group,pressure_value_group))
            #         nodeGroupInfo.extend((demand_value_group, head_value_group, pressure_value_group))
            #
            #     nodeGroupInfo.append(flow_demand)
            #     nodeGroupInfo.append(group_has_leak)
            #
            #     # print("nodeIndex : ",nodeIndex,  " - group has leak : ",group_has_leak)

            nodeIndex += 1

            # link_length = wn.query_link_attribute('length', np.less, 50)
            # print(link_length)
            # end_node_values = []
            # for linkID in link_names:
            #     link_obj = wn.get_link(linkID)
            #     # print(link_obj.start_node_name)
            #     if link_obj.start_node_name == nodeID :
            #         end_node_values.append(link_obj.end_node_name)


            hour = str(timestamp) + ":00:00"


            demand_value = demand_results.loc[hour_in_seconds,nodeID]
            # tot_junctions_demand += demand_value

            head_value = head_results.loc[hour_in_seconds, nodeID]
            head_value = "{:.8f}".format(head_value)

            pressure_value = pressure_results.loc[hour_in_seconds,nodeID]
            pressure_value = "{:.8f}".format(pressure_value)

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            leak_area_value = node_obj.leak_area  # I think that this does not require an approximation... right?
            leak_discharge_value = node_obj.leak_discharge_coeff

            leak_demand_value = leak_demand_results.loc[hour_in_seconds, nodeID]
            # tot_leaks_demand += leak_demand_value
            tot_network_demand += demand_value + leak_demand_value

            if node_type == "Junction":

                # if hasattr(node_obj, 'list_of_bds'):
                #     #happens if we set random_base_demands to True
                #     base_demand = node_obj.list_of_bds[timestamp]
                # else:
                #     # print(hour)
                #     base_demand = node_obj.demand_timeseries_list[0].base_value * \
                #                   node_obj.demand_timeseries_list[0].pattern.multipliers[int(timestamp%24)]


                list_new_demand = [0.0001, 0.0005, 0.001]
                # list_new_pattern = ["2", "9", "10"]
                # list_new_pattern = ["2", "9", "1"]
                list_new_pattern = ["2", "9", "1", "10"]
                list_new_pattern_len = [24, 24, 240, 24]

                if fixed_demand > 0:
                    #all nodes have the same base_demand and thus the same cluster
                    cluster_demand = 0
                elif fixed_demand < 0:

                    #only three different values of base_demand are considerd and thus 3 clusters
                    # if node_obj.demand_timeseries_list[0].base_value == list_new_demand[0]:
                    #     cluster_demand = 1
                    # if node_obj.demand_timeseries_list[0].base_value == list_new_demand[1]:
                    #     cluster_demand = 2
                    # if node_obj.demand_timeseries_list[0].base_value == list_new_demand[2]:
                    #     cluster_demand = 3

                    #only three different values of base_demand are considerd and thus 3 clusters
                    if node_obj.demand_timeseries_list[0].pattern.name == list_new_pattern[0]:
                        cluster_demand = 1
                        local_pattern_len = list_new_pattern_len[0]
                    if node_obj.demand_timeseries_list[0].pattern.name == list_new_pattern[1]:
                        cluster_demand = 2
                        local_pattern_len = list_new_pattern_len[1]
                    if node_obj.demand_timeseries_list[0].pattern.name == list_new_pattern[2]:
                        cluster_demand = 3
                        local_pattern_len = list_new_pattern_len[2]
                    if node_obj.demand_timeseries_list[0].pattern.name == list_new_pattern[3]:
                        cluster_demand = 4
                        local_pattern_len = list_new_pattern_len[3]


                else:
                    #no cluster because we consider base_demand origin from .inp file
                    cluster_demand = -1




                # print(hour)
                base_demand = node_obj.demand_timeseries_list[0].base_value * \
                          node_obj.demand_timeseries_list[0].pattern.multipliers[int(timestamp%local_pattern_len)]


                base_demand = "{:.8f}".format(base_demand)

                tot_juncts_demand_in_entire_simulation += demand_value
                tot_juncts_leak_demand_in_entire_simulation += leak_demand_value

            else:
                base_demand = 0.0
                demand_value = demand_value #* -1


            tot_network_demand_str = "{:.8f}".format(tot_network_demand)
            leak_demand_value = "{:.8f}".format(leak_demand_value)
            demand_value = "{:.8f}".format(demand_value)

            if (leak_area_value > 0.0):
                has_leak = True  # this leak-flag is set to true if we see a hole in the node
            else:
                has_leak = False

            tot_junctions_demand_str = "{:.8f}".format(tot_junctions_demand)
            tot_leaks_demand_str = "{:.8f}".format(tot_leaks_demand)


            out_row = [hour, nodeIdentification[nodeID], cluster_demand, base_demand, demand_value, head_value, pressure_value,
                       x_pos, y_pos, node_type, has_leak, leak_area_value,
                       leak_discharge_value, leak_demand_value,
                       # tot_junctions_demand_str, tot_leaks_demand_str, tot_network_demand_str, #end_node_values,
                       # leak_group]
                       ]


            # if node_type != "Junction":
            #     print(out_row)

            # out_row.extend(nodeGroupInfo)

            #writer.writerow(out_row)
            df.loc[len(df)] = out_row

        continue

        if True:
            for linkID in link_names:
                link_obj = wn.get_link(linkID)
                # print(linkID)
                # print(link_obj.start_node_name)
                # print(link_obj.end_node_name)
                # print(link_obj.flow)
                # link_type = link_obj.__class__.__name__

                pipeIndex += 1

                # end_node_values = []
                # for linkID in link_names:
                #     link_obj = wn.get_link(linkID)
                #     # print(link_obj.start_node_name)
                #     if link_obj.start_node_name == nodeID :
                #         end_node_values.append(link_obj.end_node_name)


                hour = str(timestamp) + ":00:00"

                flow_value = flowrate_results.loc[hour_in_seconds,linkID]
                velocity_value = velocity_results.loc[hour_in_seconds,linkID]
                status_value = status_results.loc[hour_in_seconds,linkID]

                # x_pos = node_obj.coordinates[0]
                # y_pos = node_obj.coordinates[1]


                # if node_type == "Pipe":
                #
                #     if hasattr(node_obj, 'list_of_bds'):
                #         #happens if we set random_base_demands to True
                #         base_demand = node_obj.list_of_bds[timestamp]
                #     else:
                #         base_demand = node_obj.demand_timeseries_list[0].base_value
                #
                #     base_demand = "{:.8f}".format(base_demand)
                #
                #     tot_juncts_demand_in_entire_simulation += demand_value
                #     tot_juncts_leak_demand_in_entire_simulation += leak_demand_value
                # else:
                #     base_demand = 0.0
                #     demand_value = demand_value * -1


                # out_row = [hour, linkID, link_obj.link_type, link_obj.start_node_name, link_obj.end_node_name,
                #            flow_value, velocity_value, status_value ]
                out_row = [hour, linkIdentification[linkID], link_obj.link_type, nodeIdentification[link_obj.start_node_name], nodeIdentification[link_obj.end_node_name],
                           flow_value, velocity_value, status_value ]

                #writer_link.writerow(out_row)

    #out.close()
    #print("CSV saved to: "+out_filename_complete+"\n")
    #out_link.close()
    #print("CSV saved to: "+out_filename_complete_link+"\n")


    #stats_filename = write_simulation_stats(wn, out_filename, tot_juncts_demand_in_entire_simulation, tot_juncts_leak_demand_in_entire_simulation, number_of_nodes_with_leaks, now=now)

    return df

def write_simulation_stats(wn, out_file_name, tot_nodes_demand, tot_leak_demand, number_of_nodes_with_leaks, now=None):
    print("Writing simulation stats CSV...")

    if (now == None):
        outName = out_file_name + "_nodes_simulation_stats.csv"
    else:
        outName = out_file_name + "_nodes_simulation_stats_" + now + ".csv"

    out = open(outName, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["tot_nodes_demand", "leak_percentage", "number_of_nodes", "number_of_junctions",
              "number_of_reservoirs", "number_of_tanks", "number_of_nodes_with_leaks",
              "time_spent_on_sim"]

    writer.writerow(header)

    number_of_nodes = len(wn.node_name_list)
    number_of_junctions = len(wn.junction_name_list)
    number_of_reservoirs = len(wn.reservoir_name_list)
    number_of_tanks = len(wn.tank_name_list)
    time_spent_on_sim = int(((wn.options.time.duration) / 3600)) + 1 #see in run_sim why we do +1

    if (tot_nodes_demand > 0):
        leak_percentage = (tot_leak_demand / tot_nodes_demand) * 100
        leak_percentage = round(leak_percentage, 4)
    else:
        leak_percentage = 0.0

    tot_nodes_demand = "{:.8f}".format(tot_nodes_demand)
    tot_leak_demand = "{:.8f}".format(tot_leak_demand)

    print("\nTot demand for Nodes only is: " + str(tot_nodes_demand) + " and tot_leak_demand is: " + str(
        tot_leak_demand))
    print("Total leak demand for nodes is:  " + str(leak_percentage) + "% of the Total Nodes' demand")
    print("Number of nodes inside of the network is: " + str(number_of_nodes))
    print("Number of Junctions only: " + str(number_of_junctions))
    print("Number of Reservoirs only: " + str(number_of_reservoirs))
    print("Number of Tanks only: " + str(number_of_tanks))
    print("Number of Junctions with leaks: " + str(number_of_nodes_with_leaks))
    print("Total hours simulated: " + str(time_spent_on_sim) + " (i.e. from 0:00:00 to "+str(int(time_spent_on_sim-1))+":00:00)\n")

    output_row = [tot_nodes_demand, leak_percentage, number_of_nodes, number_of_junctions,
                  number_of_reservoirs, number_of_tanks, number_of_nodes_with_leaks,
                  time_spent_on_sim]

    writer.writerow(output_row)

    out.close()

    print("Simulation stats saved to: "+outName+"\n")

    return outName

def run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename, leaks_enabled=False,
            leak_area_size=0.0000001, random_base_demands=False,  nodes_pattern="1",  patterns=None, fixed_demand=0, pattern_len=24,
            min_bd=0, max_bd=0.000005,
            min_press=0.0, req_press=0.07, file_timestamp=False, fixed_leaks=False, number_of_junctions_with_leaks=0, leak_node_array=[]):

    global list_of_fixed_nodes_with_leaks

    print("Configuring simulation...")

    complete_input_path = sim_folder_path + input_file_inp

    print("Loading INP file at: "+complete_input_path)

    wn = wntr.network.WaterNetworkModel(complete_input_path)


    # if True:
    #     node_names = wn.junction_name_list
    #
    #     for nodeID in node_names:
    #         node_obj = wn.get_node(nodeID)
    #         node_type = node_obj.__class__.__name__
    #         timestamp = 0
    #         hour = str(timestamp) + ":00:00"
    #         # x_pos = node_obj.coordinates[0]
    #         # y_pos = node_obj.coordinates[1]
    #         if node_type == "Junction":
    #
    #             if hasattr(node_obj, 'list_of_bds'):
    #                 #happens if we set random_base_demands to True
    #                 base_demand = node_obj.list_of_bds[timestamp]
    #             else:
    #                 base_demand = node_obj.demand_timeseries_list[0].base_value
    #
    #             base_demand = "{:.8f}".format(base_demand)
    #
    #         else:
    #             base_demand = 0.0
    #             demand_value = demand_value * -1




    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode
    #
    wn.options.hydraulic.minimum_pressure = min_press  # 5 psi = 3.516 m
    wn.options.hydraulic.required_pressure = req_press  # 30 psi = 21.097 m

    sim_duration_for_wntr = sim_duration - 3600

    # Why -3600? WNTR adds an hour to the sim! e.g. if we set 24 hrs it will simulate from 0:00 to 24:00 (included), so 25 hrs in total
    wn.options.time.duration = sim_duration_for_wntr

    # wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    # wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    print("Demand mode: "+str(wn.options.hydraulic.demand_model))
    print("Required pressure: "+str(wn.options.hydraulic.required_pressure))
    print("Minimum pressure: "+str(wn.options.hydraulic.minimum_pressure))
    print("Time duration (seconds): "+str(sim_duration))
    print("WNTR duration (seconds): "+str(sim_duration_for_wntr))

    if(leaks_enabled):
        print("LEAKS ARE ENABLED")

        # number_of_junctions_with_leaks = 1 #int(len(wn.junction_name_list) / 2)
        # number_of_junctions_with_leaks = int(len(wn.junction_name_list) / 4)

        # node_names = wn.junction_name_list
        # selected_junctions = pick_uniform_leaks(wn, number_of_junctions_with_leaks)
        # selected_junctions = ['8620'] #pick_rand_leaks(wn, number_of_junctions_with_leaks)
        # selected_junctions = pick_rand_leaks(wn, number_of_junctions_with_leaks)
        # selected_junctions = pick_rand_group_leaks(wn, number_of_junctions_with_leaks, leak_group, leak_node)
        selected_junctions = leak_node_array

        if(fixed_leaks):
            print("FIXED LEAKS ARE ENABLED!")
            if(list_of_fixed_nodes_with_leaks is None):
                list_of_fixed_nodes_with_leaks = selected_junctions.copy()
            selected_junctions = list_of_fixed_nodes_with_leaks

        print("FIXED LEAKS SELECTED!")
        print(selected_junctions)

        assign_leaks(wn, leak_area_size, selected_junctions)

    else:
        number_of_junctions_with_leaks = 0
        print("Leaks are NOT enabled")

    if(random_base_demands):
        print("RANDOM BASE DEMANDS ENABLED")

        results_list = execute_simulation_with_random_base_demands(wn, sim_duration_for_wntr, min_bd=min_bd, max_bd=max_bd)
        results_node, results_link = make_a_single_results_from_the_list(wn, results_list)

    else:
        print("NOT RANDOM BASE DEMANDS ENABLED")



        if  patterns is not None:

            # !VD!3#
            key = tuple(patterns)
            if key in lookup_table:
                df = lookup_table[key] #results_list = lookup_table[key]
            else:

                results_list = execute_simulation_with_custom_pattern_demands(wn, sim_duration_for_wntr, nodes_pattern, fixed_demand, patterns)

                # !VD!3#
                results_node, results_link = make_a_single_results_from_the_list(wn, results_list)
                df = write_results_to_csv(results_node, results_link, sim_duration, wn, out_filename,
                                          number_of_junctions_with_leaks, fixed_demand, pattern_len,
                                          file_timestamp=file_timestamp)

                lookup_table[key] = df
                if len(lookup_table) > MAX_LOOKUP_SIZE:
                    lookup_table.popitem(last=False)


        else:
            results_list = execute_simulation_without_random_base_demands(wn, sim_duration_for_wntr)
            #!VD!3
            results_node, results_link = make_a_single_results_from_the_list(wn, results_list)
            #!VD!3
            df = write_results_to_csv(results_node, results_link, sim_duration, wn, out_filename,
                                      number_of_junctions_with_leaks, fixed_demand, pattern_len,
                                      file_timestamp=file_timestamp)

        #!VD3!#results_node, results_link = make_a_single_results_from_the_list(wn, results_list)
        #
        # results = execute_simulation(wn)
        # results_node = results
        # results_link = results

    print("Simulation finished")

    #print(results_node.node['pressure'].columns)
    #!VD!3
    #df = write_results_to_csv(results_node, results_link, sim_duration, wn, out_filename, number_of_junctions_with_leaks, fixed_demand, pattern_len, file_timestamp=file_timestamp)

    return df

def execute_simulation(wn):
    print("\nRunning simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    return results

def make_a_single_results_from_the_list(wn, results_list):

    # fake temp class to give to the write_csv function
    class Results:
        def __init__(self, node={}):
            self.node = node

    columns = wn.node_name_list
    print(columns)
    node = OrderedDict({'demand': pd.DataFrame(columns=columns), 'head': pd.DataFrame(columns=columns), 'pressure': pd.DataFrame(columns=columns), 'leak_demand': pd.DataFrame(columns=columns)})

    columns_link = wn.link_name_list
    print(columns_link)
    link = OrderedDict({'flowrate': pd.DataFrame(columns=columns_link), 'velocity': pd.DataFrame(columns=columns_link),
                        'status': pd.DataFrame(columns=columns_link)})

    for results in results_list:

        demand_results = results.node['demand']
        head_results = results.node['head']
        pressure_results = results.node['pressure']
        leak_demand_results = results.node["leak_demand"]

        flow_results = results.link['flowrate']
        velocity_results = results.link['velocity']
        status_results = results.link['status']

        # merge the two dataframes vertically (row-wise) using pd.concat
        node['demand'] = pd.concat([node['demand'], demand_results], axis=0)
        node['head'] = pd.concat([node['head'], head_results], axis=0)
        node['pressure'] = pd.concat([node['pressure'], pressure_results], axis=0)
        node['leak_demand'] = pd.concat([node['leak_demand'], leak_demand_results], axis=0)

        link['flowrate'] = pd.concat([link['flowrate'], flow_results], axis=0)
        link['velocity'] = pd.concat([link['velocity'], velocity_results], axis=0)
        link['status'] = pd.concat([link['status'], status_results], axis=0)

    out_node = Results(node)
    out_link = Results(link)

    return out_node, out_link

def execute_simulation_with_custom_pattern_demands(wn, sim_duration_for_wntr, nodes_pattern, fixed_demand, patterns=None, min_bd=0, max_bd=0.000005):
    print("\nRunning simulation... (with_custom_pattern_demands)")
    #pattern = create_custom_pattern(wn,"custom",1,1,1, sim_duration_for_wntr)

    sim_duration_in_hours = int(sim_duration_for_wntr / 3600) + 1
    node_names = wn.junction_name_list
    print("***** execute_simulation_with_custom_pattern_demands **** NUM NODE : ")
    print(len(node_names))

    results_list = []
    for hour in range(sim_duration_in_hours):
        wn.options.time.duration = hour * 3600

        index_new_demand = 0
        # list_new_demand = [0.5, 1, 1.5]
        # list_new_demand = [0.1, 0.03, 0.01] # [0.0001, 0.0005, 0.001]
        list_new_demand = [0.1, 0.015, 0.01] # [0.0001, 0.0005, 0.001]

        # nodes_pattern_array = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        # pattern_len_array = [240, 240, 24, 240, 240, 240, 936, 168, 168, 24, 24]
        index_new_pattern = 0

        # list_new_pattern = ["2", "9", "10"] #attenzione alla lunghezza che serve sistemarla sotto, in questo caso hanno tutti la stessa lunghezza
        # list_new_pattern = ["2", "9", "1"] #attenzione alla lunghezza che serve sistemarla sotto, in questo caso hanno tutti la stessa lunghezza
        list_new_pattern = ["2", "9", "1", "10"] #attenzione alla lunghezza che serve sistemarla sotto, in questo caso hanno tutti la stessa lunghezza

        #patterns = pd.read_csv("node_patterns.csv")["cluster_demand"].to_numpy()
        
        for juncID in node_names:
            junc_obj = wn.get_node(juncID)

            if fixed_demand > 0: #FIXED
                new_demand = fixed_demand
                new_pattern = nodes_pattern

            elif fixed_demand < 0: #CLUSTER


                new_demand = list_new_demand[1]
                # new_demand = list_new_demand[index_new_demand%3]
                # index_new_demand += 1

                # FILIPPO
                # modificare questo per assegnare in modo statico un pattern a un certo device
                # tu ti crei un file csv con la lista dei device, nella seconda colonna il cluster di appartenenza index cluster [0, 1, 2]
                # modificando il file csv sei in grado di fissare per ogni device il cluster che gli vuoi assegnare
                # in questo punto il codice legge il file e assegna new_pattern in funzione dell'indice, cioè index_new_pattern lo leggi dal file e può valere 0, 1, 2
                # all'inizio metti tutti i nodi sul cluster 0, e uno solo sul cluster 1 --> nodo di riferimento
                # fai 3  simulazioni in cui il nodo di riferimento cambia cluster
                # la durata è di 96 ore
                #print("index:", index_new_pattern)

                if patterns[index_new_pattern] == -1:
                    index_new_pattern += 1
                    continue

                new_pattern = list_new_pattern[patterns[index_new_pattern]] # list_new_pattern[index_new_pattern%3]
                index_new_pattern += 1

                # index_from_rand = random.randint(0,2)
                # new_demand = list_new_demand[int(index_from_rand)]

            else:   #ORIGIN
                new_demand = junc_obj.base_demand
                new_pattern = nodes_pattern

            # junc_obj.demand_timeseries_list[0].pattern = pattern

            junc_obj.add_demand(base=new_demand, pattern_name=new_pattern)
            del junc_obj.demand_timeseries_list[0]

            # To keep track of the history of the different random base demands we add a custom field to the Junction object of wntr
            # if it is the first time that we call this, we have to create it, else we just append the new value to the list
            if hasattr(junc_obj, 'list_of_bds'):
                junc_obj.list_of_bds.append(junc_obj.base_demand)
            else:
                junc_obj.list_of_bds = [junc_obj.base_demand]

        # wntr.network.write_inpfile(wn, 'domenico.inp', version=2.2)
        # exit(1)

        results = wntr.sim.WNTRSimulator(wn).run_sim()

        results_list.append(results)

    return results_list



def execute_simulation_with_random_base_demands(wn, sim_duration_for_wntr, min_bd=0, max_bd=0.000005):
    print("\nRunning simulation... (with_random_base_demands)")
    pattern = create_custom_pattern(wn,"custom",1,1,1, sim_duration_for_wntr)

    sim_duration_in_hours = int(sim_duration_for_wntr / 3600) + 1

    results_list = []

    for hour in range(sim_duration_in_hours):

        wn.options.time.duration = hour * 3600

        assign_rand_demand_to_junctions(wn, min_bd, max_bd, "custom")

        # wntr.network.write_inpfile(wn, 'one5.inp', version=2.2)
        # exit(1)

        results = wntr.sim.WNTRSimulator(wn).run_sim()

        results_list.append(results)

    return results_list

def execute_simulation_without_random_base_demands(wn, sim_duration_for_wntr):
    print("\nRunning simulation... (without_random_base_demands)")

    sim_duration_in_hours = int(sim_duration_for_wntr / 3600) + 1

    results_list = []

    for hour in range(sim_duration_in_hours):

        wn.options.time.duration = hour * 3600

        results = wntr.sim.WNTRSimulator(wn).run_sim()

        results_list.append(results)

    return results_list



def merge_multiple_datasets(exported_path, datasets_to_merge, output_filename, delete_old_files=False):
    print("Merging CSVs...")

    if(delete_old_files):
        print("Deletion of old unmerged CSVs ENABLED!")
    else:
        print("Deletion of old unmerged CSVs NOT enabled")

    print("Merging these datasets:")

    output_filename_merge = exported_path + output_filename + "_nodes.csv"

    pd.options.display.float_format = '{:,.8f}'.format

    path_to_first_df = datasets_to_merge.pop(0)

    print(path_to_first_df)

    # We read our entire dataset
    first_df = pd.read_csv(path_to_first_df, delimiter=';')

    last_row_from_first_df = first_df.iloc[-1]["hour"]

    # cheap hack useful to know how many nodes we have in a dataset
    number_of_nodes = first_df['hour'].value_counts()[last_row_from_first_df]

    last_hour_from_first_df = int(last_row_from_first_df.split(":",1)[0]) + 1

    while len(datasets_to_merge) > 0:
        data_path = datasets_to_merge.pop(0)
        print(data_path)

        next_df = pd.read_csv(data_path, header=0, delimiter=';')

        n_iterations = int(len(next_df) / number_of_nodes)

        for mult in range(1, n_iterations+1):
            stop = mult * number_of_nodes
            start = stop - number_of_nodes
            value = str(last_hour_from_first_df)+":00:00"
            next_df.loc[start:stop, 'hour'] = value

            last_hour_from_first_df += 1

        first_df = pd.concat([first_df, next_df], ignore_index=True)

        if(delete_old_files):
            if os.path.exists(data_path):
                os.remove(data_path)
            else:
                print("Deletion NOT successful!: "+data_path)

    #first_df.to_csv(output_filename_merge, float_format='%.8f', index=False, sep=';', decimal= ",")
    first_df.to_csv(output_filename_merge, float_format='%.8f', index=False, sep=';')

    if (delete_old_files):
        if os.path.exists(path_to_first_df):
            os.remove(path_to_first_df)
        else:
            print("Deletion NOT successful!: "+path_to_first_df)

    print()
    print("Merge finished. Final csv saved to: "+output_filename_merge)



def merge_multiple_datasets_link(exported_path, datasets_to_merge, output_filename, delete_old_files=False):
    print("Merging CSVs...")

    if(delete_old_files):
        print("Deletion of old unmerged CSVs ENABLED!")
    else:
        print("Deletion of old unmerged CSVs NOT enabled")

    print("Merging these datasets:")

    output_filename_merge = exported_path + output_filename + "_links.csv"

    pd.options.display.float_format = '{:,.8f}'.format

    path_to_first_df = datasets_to_merge.pop(0)

    print(path_to_first_df)

    # We read our entire dataset
    first_df = pd.read_csv(path_to_first_df, delimiter=';')

    last_row_from_first_df = first_df.iloc[-1]["hour"]

    # cheap hack useful to know how many nodes we have in a dataset
    number_of_links = first_df['hour'].value_counts()[last_row_from_first_df]

    last_hour_from_first_df = int(last_row_from_first_df.split(":",1)[0]) + 1

    while len(datasets_to_merge) > 0:
        data_path = datasets_to_merge.pop(0)
        print(data_path)

        next_df = pd.read_csv(data_path, header=0, delimiter=';')

        n_iterations = int(len(next_df) / number_of_links)

        for mult in range(1, n_iterations+1):
            stop = mult * number_of_links
            start = stop - number_of_links
            value = str(last_hour_from_first_df)+":00:00"
            next_df.loc[start:stop, 'hour'] = value

            last_hour_from_first_df += 1

        first_df = pd.concat([first_df, next_df], ignore_index=True)

        if(delete_old_files):
            if os.path.exists(data_path):
                os.remove(data_path)
            else:
                print("Deletion NOT successful!: "+data_path)

    #first_df.to_csv(output_filename_merge, float_format='%.8f', index=False, sep=';', decimal= ",")
    first_df.to_csv(output_filename_merge, float_format='%.8f', index=False, sep=';')

    if (delete_old_files):
        if os.path.exists(path_to_first_df):
            os.remove(path_to_first_df)
        else:
            print("Deletion NOT successful!: "+path_to_first_df)

    print()
    print("Merge finished. Final csv saved to: "+output_filename_merge)



def merge_multiple_stats(exported_path, stats_to_merge, out_filename, delete_old_files=False):
    print("merge_multiple_stats is currently UNSUPPORTED! (will delete old files anyway...)")

    for path in stats_to_merge:

        if (delete_old_files):
            if os.path.exists(path):
                os.remove(path)
            else:
                print("Deletion NOT successful!: " + path)

def run_multiple_sims(exported_path, sim_folder_path, input_file_inp, sim_duration, out_filename, number_of_sims,
                      leaks_enabled=False, leak_area_size=0.0000001, random_base_demands=False,  nodes_pattern="1", patterns=None, fixed_demand=0, pattern_len=24,
                      min_bd=0, max_bd=0.000005, min_press=0.0, req_press=0.07, file_timestamp=False,
                      delete_old_files=False, merge_csv=True, fixed_leaks=False, number_of_junctions_with_leaks=0, leak_node_array=[]):
    
    results_from_sim = run_sim(sim_folder_path, input_file_inp, sim_duration,
                                   out_filename, leaks_enabled=leaks_enabled,
                                   leak_area_size=leak_area_size,
                                   random_base_demands=random_base_demands,
                                   nodes_pattern=nodes_pattern, patterns=patterns, fixed_demand=fixed_demand, pattern_len=pattern_len,
                                   min_bd=min_bd, max_bd=max_bd, min_press=min_press,
                                   req_press=req_press, file_timestamp=file_timestamp,
                                   fixed_leaks=fixed_leaks, number_of_junctions_with_leaks=number_of_junctions_with_leaks, leak_node_array=leak_node_array)
    
    print(results_from_sim["cluster_demand"])

def simulation_step(patterns):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    #df = run_sim("", "NET_30_pattern_single_peak.inp", 24*3600,
    # df=run_sim("", "NET_30_pattern.inp", 24 * 3600,
    df = run_sim("", "NET_30_pattern_single_peak_no_default_pattern.inp", 24*3600,
                                "simulation", leaks_enabled=False,
            leak_area_size=0.0002 * 82 * 1.5,
            random_base_demands=False,
            nodes_pattern=0, patterns=patterns, fixed_demand=-1, pattern_len=240,
            min_bd=0, max_bd=0.005, min_press=3.516,
            req_press=21.097, file_timestamp=True,
            fixed_leaks=True, number_of_junctions_with_leaks=5, leak_node_array=[ "8626",  "8684" , "8740"])
    
    sys.stdout.close()
    sys.stdout = original_stdout
    
    return df

def get_nodes():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # Suppress prints

    #df = run_sim("", "NET_30_pattern.inp", 24*3600,
    df = run_sim("", "NET_30_pattern_single_peak_no_default_pattern.inp", 24*3600,
            "simulation", leaks_enabled=False,
            leak_area_size=0.0002 * 82 * 1.5,
            random_base_demands=False,
            nodes_pattern=0, patterns=None, fixed_demand=-1, pattern_len=240,
            min_bd=0, max_bd=0.005, min_press=3.516,
            req_press=21.097, file_timestamp=True,
            fixed_leaks=True, number_of_junctions_with_leaks=5, leak_node_array=[ "8626",  "8684" , "8740"])
    
    sys.stdout.close()
    sys.stdout = original_stdout # Restore prints
    
    return df["nodeID"].unique()

if __name__ == "__main__":
    print("******   py_epanet started!  ******\n")

    ############## MAIN input FOLDER
    # sim_folder_path = "./networks/other-network/"
    # input_file_inp = "month_large_run.inp"
    sim_folder_path = ""

    ############## WATER NETWORK
    # input_file_inp = "NET_0.inp"
    # input_file_inp = "NET_1.inp" # input_file_inp = "NET_1_v2.inp"
    # input_file_inp = "NET_2.inp"; net_directory = 'NET_2/'

    # input_file_inp = "NET_3.inp"; net_directory = 'NET_3/'
    #input_file_inp = "NET_30_pattern.inp"; net_directory = 'NET_3_pattern/'
    input_file_inp = "NET_30_pattern_single_peak_no_default_pattern.inp"; net_directory = 'NET_3_pattern/'
    # input_file_inp = "NET_4.inp"; net_directory = 'NET_4/'  --> non funzionante
    # input_file_inp = "NET_6_fixed.inp"; net_directory = 'NET_6/'

    ############## DURATION
    sim_duration_h = 24; h_directory = 'h_24/'  # 720 #24
    # sim_duration_h = 24; h_directory = 'h_24/' # # 720 #24

    ############## MAIN OUTPUT FOLDER
    exported_path = ''

    ############## LEAK
    leaks_enabled = False  # switch this to True to enable leaks assignments
    number_of_junctions_with_leaks = 5
#    leak_node_array = [ "8626",  "8660" , "8684" , "8706", "8740"] #net3
    leak_node_array = [ "8626",  "8684" , "8740"] #net3
    #leak_node_array = [ "J-155",  "J-339" , "J-253" , "J-296", "J-683"]
    #leak_node_array = [ "138",  "133" , "168" , "140", "176", "732"] #net6
    leak_area_array = [ "000205",  "00082" , "0164" , "0246"]
    leak_area = leak_area_array[3]
    fixed_leaks = True  # switch this to True to have the random picks for nodes executed only once in multiple sims

    ############## BASE DEMAND
    random_base_demands = False  # False  # switch this to True to enable random base demand assignments
    min_bd = 0  # minimum possible random base demand
    max_bd = 0.005 #0.05 #0.0001 #0.005  # maximum possible random base demand

    nodes_pattern_array = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    pattern_len_array = [240, 240, 24, 240, 240, 240, 936, 168, 168, 24, 24]
    fixed_demand_arry = [-1, 0, 1] # [cluster, origin, fixed]
    fixed_demand_str_array = ["cluster", "origin", "1"]

    nodes_pattern = nodes_pattern_array[1]
    pattern_len = pattern_len_array[1]
    fixed_demand = fixed_demand_arry[0]
    fixed_demand_str = fixed_demand_str_array[0]

    for leak_group_index in range(1,2,1):
        for leak_node_index in range(10,11,1):
    # for leak_group_index in range(3, 8, 1):
    #     for leak_node_index in range(5, 10, 1):

            leak_group = leak_group_index
            leak_node = leak_node_index

            if leak_area == "000205":
                leak_area_size = 0.0002 * 82 / 80  # 0.0082  # area of the "hole" of the leak
            elif leak_area == "00082":
                leak_area_size = 0.0002 * 82 / 20  # 0.0082  # area of the "hole" of the leak
            elif leak_area == "0164":
                leak_area_size = 0.0002 * 82 / 1  # 0.0082  # area of the "hole" of the leak
            elif leak_area=="0246":
                leak_area_size = 0.0002 * 82 * 1.5  # 0.082  # area of the "hole" of the leak
            else:
                print("select leak area")
                sys.exit(1)

            print("leak area : " + str(leak_area_size))
            # sys.exit(1)
            sim_duration = sim_duration_h * 3600  # hours in seconds

            out_filename = "simulation"

            print(out_filename)

            min_press = 3.516   # 5 psi = 3.516 m
            req_press = 21.097  # 30 psi = 21.097 m

            file_timestamp = True  # switch this to True to write a current timestamp to the output filename

            merge_csv = True  # switch this to True to merge CSVs into one
            delete_old_files = True  # switch this to True to delete old unmerged CSVs after merging them into one

            number_of_sims = 1 #i * 7 * 4
            run_multiple_sims(exported_path, sim_folder_path, input_file_inp, sim_duration, out_filename, number_of_sims,
                                  leaks_enabled=leaks_enabled, leak_area_size=leak_area_size,
                                  random_base_demands=random_base_demands,
                                  nodes_pattern=nodes_pattern, fixed_demand=fixed_demand, pattern_len=pattern_len,
                                  min_bd=min_bd, max_bd=max_bd, min_press=min_press, req_press=req_press,
                                  file_timestamp=file_timestamp, delete_old_files=delete_old_files, merge_csv=merge_csv,
                                  fixed_leaks=fixed_leaks, number_of_junctions_with_leaks=number_of_junctions_with_leaks, leak_node_array=leak_node_array)

    print("\nExiting...")