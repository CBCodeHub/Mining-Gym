"""
Snapshot-style KPI calculator. Each call overwrites the previous JSON
output (current-state snapshot, not history). For append-only per-episode
history, see episode_metrics_logger.py.
"""

import json


class KPI_Calculator:
    def __init__(self):
        self.previous_num_trips = 0
        self.previous_avg_idle_time = 0

    def calculate_kpis(self, num_trips, material_load_per_trip, total_known_cost,
                        total_estimated_cost, fuel_per_trip, total_downtime,
                        avg_idle_time, num_trucks, shift_duration, upload_counter,
                        resource_status):
        delta_num_trips = num_trips - self.previous_num_trips
        delta_avg_idle_time = avg_idle_time - self.previous_avg_idle_time

        total_production = delta_num_trips * material_load_per_trip

        equipment_utilization = ((shift_duration - total_downtime - delta_avg_idle_time)
                                  / (shift_duration - total_downtime)) * 100

        if total_production == 0:
            cost_per_ton = 0
        else:
            cost_per_ton = (total_known_cost + total_estimated_cost) / total_production

        total_fuel_consumed = delta_num_trips * fuel_per_trip
        if total_production == 0:
            fuel_consumption = 0
        else:
            fuel_consumption = total_fuel_consumed / total_production

        kpis = {
            'Total Production (PVOL)': total_production,
            'Equipment Utilization (EUSE)': equipment_utilization,
            'Cost per Ton (CPT)': cost_per_ton,
            'Fuel Consumption (FC)': fuel_consumption,
            "upload_counter": upload_counter,
            "Number of trips": delta_num_trips,
            "Avg. idle time": delta_avg_idle_time,
            "Total fuel consumed": total_fuel_consumed,
        }

        return kpis

    def save_kpis_to_json(self, kpis, filename='kpi_results.json'):
        with open(filename, 'w') as file:
            json.dump(kpis, file, indent=4)

    def calculate_and_save_kpis(self, num_trips, material_load_per_trip, total_known_cost,
                                 total_estimated_cost, fuel_per_trip, total_downtime,
                                 avg_idle_time, num_trucks, shift_duration, upload_counter,
                                 resource_status, filename='kpi_results.json'):
        kpis = self.calculate_kpis(num_trips, material_load_per_trip, total_known_cost,
                                    total_estimated_cost, fuel_per_trip, total_downtime,
                                    avg_idle_time, num_trucks, shift_duration, upload_counter,
                                    resource_status)
        self.save_kpis_to_json(kpis, filename)

        self.previous_num_trips = num_trips
        self.previous_avg_idle_time = avg_idle_time

    def update_resource_status(self, resource_status, filename='kpi_results.json'):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        data.update({
            "Status of Shovel": resource_status['Shovels'],
        })

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
