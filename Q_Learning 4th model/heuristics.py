# import libraries
import numpy as np
import pandas as pd
import random
import re
import datetime


def longest_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            total += process_t[Job, Machine]
        totalProcessingTime.append(total)

    return sorted(range(len(totalProcessingTime)), reverse=True, key=lambda k: totalProcessingTime[k])


def shortest_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            total += process_t[Job, Machine]
        totalProcessingTime.append(total)

    return sorted(range(len(totalProcessingTime)), reverse=False, key=lambda k: totalProcessingTime[k])


def longest_weighted_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            total += (n_mach-Machine) * process_t[Job, Machine]
        totalProcessingTime.append(total)

    return sorted(range(len(totalProcessingTime)), reverse=True, key=lambda k: totalProcessingTime[k])


def shortest_weighted_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            total += (n_mach-Machine) * process_t[Job, Machine]
        totalProcessingTime.append(total)

    return sorted(range(len(totalProcessingTime)), reverse=False, key=lambda k: totalProcessingTime[k])


def shortest_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            for Job2 in range(0, n_jobs):
                if Job2 != Job:
                    total += abs(process_t[Job, Machine] - process_t[Job2, Machine])
        totalProcessingTime.append(total)
    return sorted(range(len(totalProcessingTime)), reverse=False, key=lambda k: totalProcessingTime[k])


def longest_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            for Job2 in range(0, n_jobs):
                if Job2 != Job:
                    total += abs(process_t[Job, Machine] - process_t[Job2, Machine])
        totalProcessingTime.append(total)
    return sorted(range(len(totalProcessingTime)), reverse=True, key=lambda k: totalProcessingTime[k])


def shortest_weighted_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            for Job2 in range(0, n_jobs):
                if Job2 != Job:
                    total += (n_mach-Machine) * abs(process_t[Job, Machine] - process_t[Job2, Machine])
        totalProcessingTime.append(total)
    return sorted(range(len(totalProcessingTime)), reverse=False, key=lambda k: totalProcessingTime[k])


def longest_weighted_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach):
    totalProcessingTime = []
    for Job in range(0, n_jobs):
        total = 0
        for Machine in range(0, n_mach):
            for Job2 in range(0, n_jobs):
                if Job2 != Job:
                    total += (n_mach-Machine) * abs(process_t[Job, Machine] - process_t[Job2, Machine])
        totalProcessingTime.append(total)
    return sorted(range(len(totalProcessingTime)), reverse=True, key=lambda k: totalProcessingTime[k])


def heuristics(process_t, n_jobs, n_mach):
    initial_sol1 = shortest_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol2 = longest_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol3 = shortest_weighted_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol4 = longest_weighted_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol5 = shortest_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol6 = longest_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol7 = shortest_weighted_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol8 = longest_weighted_abs_total_processing_time_heuristic(process_t, n_jobs, n_mach)
    initial_sol9 = list(range(0, n_jobs)) ## random solution
    random.shuffle(initial_sol9)

    return initial_sol1, initial_sol2, initial_sol3, initial_sol4, initial_sol5, initial_sol6, initial_sol7, initial_sol8, initial_sol9

