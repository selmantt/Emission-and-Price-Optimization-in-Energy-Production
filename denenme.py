import pandas as pd
import numpy as np
import random
import io

# Sağlanan CSV verisi
csv_data_string = """YEAR,CO2_Emissions_TargetSector_Tonne,AvgHouseholdPrice_Euro_per_kWh,TotalProduction_GWh,RenewableProduction_GWh,RenewableShare_Percent,Anthracite_Share_Percent,Batteries_Share_Percent,Bioenergy_Share_Percent,Biogases_Share_Percent,Blast_furnace_gas_Share_Percent,Brown_coal_briquettes_Share_Percent,Coke_oven_coke_Share_Percent,Coke_oven_gas_Share_Percent,Coking_coal_Share_Percent,Fossil_energy_Share_Percent,Fuel_oil_Share_Percent,Gas_oil_and_diesel_oil_(excluding_biofuel_portion)_Share_Percent,Geothermal_Share_Percent,Hydro_Share_Percent,Industrial_waste_(non-renewable)_Share_Percent,Lignite_Share_Percent,Liquefied_petroleum_gases_Share_Percent,Manufactured_gases_Share_Percent,Natural_gas_Share_Percent,Non-renewable_municipal_waste_Share_Percent,Non-renewable_waste_Share_Percent,Nuclear_heat_Share_Percent,Oil_and_petroleum_products_(excluding_biofuel_portion)_Share_Percent,Other_bituminous_coal_Share_Percent,Other_fuels_n.e.c._Share_Percent,Other_liquid_biofuels_Share_Percent,Other_oil_products_n.e.c._Share_Percent,Other_recovered_gases_Share_Percent,Petroleum_coke_Share_Percent,Primary_solid_biofuels_Share_Percent,Pumped_hydro_power_Share_Percent,Refinery_gas_Share_Percent,Renewable_municipal_waste_Share_Percent,Renewables_and_biofuels_Share_Percent,Solar_photovoltaic_Share_Percent,Solid_fossil_fuels_Share_Percent,Wind_Share_Percent
2023,171449422.01,0.40725,1202149.925,558511.0,46.45934657442997,0.01855009889885407,0.01538094343764984,3.9134377519509473,2.5040137984453144,0.5853679190638389,0.04076030699748203,0.0006654743999588903,0.16129435769003603,0.3987023498753702,18.835001133490067,0.07744458329521585,0.06954207479570404,0.01622093849899795,1.6548684640977707,0.05989269599630013,7.136880202359119,0.030528638098114094,0.8251882559490239,6.727530262084406,0.4744832471706888,0.5343759431669889,0.6002579087629191,0.4468660595723949,2.800178272273319,0.1134633851929908,0.008651167199465574,0.20812711858714295,0.07852597919514906,0.009150272999434743,0.8315934470486283,0.45277214487203,0.05207337179678317,0.4744832471706888,22.468911271611983,5.2885250564733015,10.395736704804103,11.690555152677815
"""
df = pd.read_csv(io.StringIO(csv_data_string))
latest_year_data = df[df['YEAR'] == 2023].iloc[0]

x0 = latest_year_data['AvgHouseholdPrice_Euro_per_kWh']
y0_percent = latest_year_data['RenewableShare_Percent']
y0 = y0_percent / 100.0
z0 = latest_year_data['CO2_Emissions_TargetSector_Tonne']
T0 = latest_year_data['TotalProduction_GWh']

print(f"Başlangıç Değerleri (2023):")
print(f"  x0 (Ort. Elektrik Fiyatı): {x0:.4f} Euro/kWh")
print(f"  y0 (Yenilenebilir Payı - Orijinal): {y0:.4f} (%{y0_percent:.2f})")
print(f"  z0 (Toplam CO2 Emisyonu): {z0:,.0f} Tonne")
print(f"  T0 (Toplam Üretim): {T0:,.0f} GWh")

share_columns = [
    'Anthracite_Share_Percent', 'Bioenergy_Share_Percent', 'Biogases_Share_Percent',
    'Blast_furnace_gas_Share_Percent', 'Brown_coal_briquettes_Share_Percent', 'Coke_oven_coke_Share_Percent',
    'Coke_oven_gas_Share_Percent', 'Coking_coal_Share_Percent', 'Fuel_oil_Share_Percent',
    'Gas_oil_and_diesel_oil_(excluding_biofuel_portion)_Share_Percent', 'Geothermal_Share_Percent',
    'Hydro_Share_Percent', 'Industrial_waste_(non-renewable)_Share_Percent', 'Lignite_Share_Percent',
    'Liquefied_petroleum_gases_Share_Percent', 'Manufactured_gases_Share_Percent', 'Natural_gas_Share_Percent',
    'Non-renewable_municipal_waste_Share_Percent', 'Nuclear_heat_Share_Percent', 'Other_bituminous_coal_Share_Percent',
    'Other_fuels_n.e.c._Share_Percent', 'Other_liquid_biofuels_Share_Percent', 'Other_oil_products_n.e.c._Share_Percent',
    'Other_recovered_gases_Share_Percent', 'Petroleum_coke_Share_Percent', 'Primary_solid_biofuels_Share_Percent',
    'Pumped_hydro_power_Share_Percent', 'Refinery_gas_Share_Percent', 'Renewable_municipal_waste_Share_Percent',
    'Solar_photovoltaic_Share_Percent', 'Wind_Share_Percent'
]
initial_shares_dict = {col: latest_year_data.get(col, 0) / 100.0 for col in share_columns}
num_decision_variables = len(share_columns)
print(f"Karar değişkeni sayısı: {num_decision_variables}")
print(f"  GA'nın referans alacağı başlangıç paylarının toplamı: {sum(initial_shares_dict.values()):.4f}")

renewable_sources_for_y = [
    'Bioenergy_Share_Percent', 'Biogases_Share_Percent', 'Geothermal_Share_Percent', 'Hydro_Share_Percent',
    'Other_liquid_biofuels_Share_Percent', 'Primary_solid_biofuels_Share_Percent',
    'Renewable_municipal_waste_Share_Percent', 'Solar_photovoltaic_Share_Percent', 'Wind_Share_Percent'
]
low_carbon_sources_zero_emission_assumption = ['Nuclear_heat_Share_Percent', 'Pumped_hydro_power_Share_Percent']
fossil_sources_for_z = [
    s for s in share_columns
    if s not in renewable_sources_for_y and s not in low_carbon_sources_zero_emission_assumption
]
initial_fossil_share_for_z_calc = sum(initial_shares_dict.get(s, 0) for s in fossil_sources_for_z)
print(f"  z_new hesaplaması için başlangıç fosil payı: {initial_fossil_share_for_z_calc:.4f}")

unit_costs_euro_per_kwh = {
    'Solar_photovoltaic_Share_Percent': 0.045, 'Wind_Share_Percent': 0.06, 'Bioenergy_Share_Percent': 0.11,
    'Biogases_Share_Percent': 0.13, 'Hydro_Share_Percent': 0.045, 'Geothermal_Share_Percent': 0.09,
    'Renewable_municipal_waste_Share_Percent': 0.075, 'Other_liquid_biofuels_Share_Percent': 0.15,
    'Primary_solid_biofuels_Share_Percent': 0.11, 'Natural_gas_Share_Percent': 0.10,
    'Lignite_Share_Percent': 0.08, 'Anthracite_Share_Percent': 0.09, 'Other_bituminous_coal_Share_Percent': 0.09,
    'Coking_coal_Share_Percent': 0.09, 'Brown_coal_briquettes_Share_Percent': 0.08,
    'Coke_oven_coke_Share_Percent': 0.09, 'Nuclear_heat_Share_Percent': 0.04, 'Fuel_oil_Share_Percent': 0.18,
    'Gas_oil_and_diesel_oil_(excluding_biofuel_portion)_Share_Percent': 0.20,
    'Liquefied_petroleum_gases_Share_Percent': 0.15, 'Other_oil_products_n.e.c._Share_Percent': 0.20,
    'Petroleum_coke_Share_Percent': 0.10, 'Blast_furnace_gas_Share_Percent': 0.03,
    'Coke_oven_gas_Share_Percent': 0.03, 'Manufactured_gases_Share_Percent': 0.05,
    'Industrial_waste_(non-renewable)_Share_Percent': 0.06, 'Non-renewable_municipal_waste_Share_Percent': 0.06,
    'Other_recovered_gases_Share_Percent': 0.04, 'Refinery_gas_Share_Percent': 0.04,
    'Other_fuels_n.e.c._Share_Percent': 0.10, 'Pumped_hydro_power_Share_Percent': 0.02
}
default_cost = 0.12
for source_key in share_columns:
    if source_key not in unit_costs_euro_per_kwh: unit_costs_euro_per_kwh[source_key] = default_cost

installation_costs_euro_per_kwh_increase = {}
HIGH_INSTALL_COST = 25.0
VERY_HIGH_INSTALL_COST_FOR_NICHE = 35.0
def calculate_amortized_capex(capex_eur_per_kw, cap_factor, lt_years, risk_premium=1.0):
    if cap_factor <= 0 or lt_years <= 0: return HIGH_INSTALL_COST
    return (capex_eur_per_kw * risk_premium) / (cap_factor * 8760 * lt_years)

installation_costs_euro_per_kwh_increase.update({
    'Solar_photovoltaic_Share_Percent': calculate_amortized_capex(1500, 0.10, 25, 1.15), # ~0.046
    'Wind_Share_Percent': calculate_amortized_capex(1900, 0.22, 30, 1.15),  # ~0.043
    'Bioenergy_Share_Percent': calculate_amortized_capex(4800, 0.55, 18, 1.25), # ~0.070
    'Biogases_Share_Percent': calculate_amortized_capex(5200, 0.55, 18, 1.25), # ~0.076
    'Primary_solid_biofuels_Share_Percent': calculate_amortized_capex(4800, 0.55, 18, 1.25), # ~0.070
    'Renewable_municipal_waste_Share_Percent': calculate_amortized_capex(4800, 0.55, 18, 1.25), # ~0.070
    'Non-renewable_municipal_waste_Share_Percent': calculate_amortized_capex(4800, 0.55, 18, 1.25),
    'Hydro_Share_Percent': calculate_amortized_capex(6000, 0.28, 35, 1.25), # ~0.087
    'Geothermal_Share_Percent': calculate_amortized_capex(8000, 0.55, 25, 1.35), # ~0.099
    'Other_liquid_biofuels_Share_Percent': calculate_amortized_capex(6000, 0.45, 15, 1.35), # ~0.123
    'Pumped_hydro_power_Share_Percent': calculate_amortized_capex(4000, 0.08, 10, 1.3), # ~0.185
    'Natural_gas_Share_Percent': calculate_amortized_capex(1100, 0.40, 30, 1.1), # ~0.010
    'Lignite_Share_Percent': HIGH_INSTALL_COST, 'Anthracite_Share_Percent': HIGH_INSTALL_COST,
    'Other_bituminous_coal_Share_Percent': HIGH_INSTALL_COST, 'Coking_coal_Share_Percent': HIGH_INSTALL_COST,
    'Brown_coal_briquettes_Share_Percent': HIGH_INSTALL_COST, 'Coke_oven_coke_Share_Percent': HIGH_INSTALL_COST,
    'Nuclear_heat_Share_Percent': VERY_HIGH_INSTALL_COST_FOR_NICHE, 'Fuel_oil_Share_Percent': HIGH_INSTALL_COST,
    'Gas_oil_and_diesel_oil_(excluding_biofuel_portion)_Share_Percent': HIGH_INSTALL_COST,
    'Liquefied_petroleum_gases_Share_Percent': HIGH_INSTALL_COST, 'Other_oil_products_n.e.c._Share_Percent': HIGH_INSTALL_COST,
    'Petroleum_coke_Share_Percent': HIGH_INSTALL_COST, 'Blast_furnace_gas_Share_Percent': HIGH_INSTALL_COST,
    'Coke_oven_gas_Share_Percent': HIGH_INSTALL_COST, 'Manufactured_gases_Share_Percent': HIGH_INSTALL_COST,
    'Industrial_waste_(non-renewable)_Share_Percent': HIGH_INSTALL_COST, 'Other_recovered_gases_Share_Percent': HIGH_INSTALL_COST,
    'Refinery_gas_Share_Percent': HIGH_INSTALL_COST, 'Other_fuels_n.e.c._Share_Percent': HIGH_INSTALL_COST
})
default_install_cost = HIGH_INSTALL_COST
for skey in share_columns:
    if skey not in installation_costs_euro_per_kwh_increase: installation_costs_euro_per_kwh_increase[skey] = default_install_cost

print("\nBirim Çalıştırma ve Kurulum Maliyetleri (AGRESİF VE MUHAFAZAKAR - SON TUR v3) tanımlandı.")

POPULATION_SIZE = 250; MAX_GENERATIONS = 400; MUTATION_RATE = 0.08; CROSSOVER_RATE = 0.85
ALPHA = 3.0; BETA = 0.8; GAMMA = 0.3
TARGET_PRICE_INCREASE_MAX = 1.1
TARGET_EMISSION_REDUCTION_MIN = 0.95
MAX_ABSOLUTE_CHANGE_PUNITS = 0.015
ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES = 0.025 # İsteğiniz üzerine %2.5 puan
PENALTY_WEIGHT_SHARE_CHANGE = 750000 # Ceza AĞIRLIĞI MAKSİMUM +
EXTRA_PENALTY_FACTOR_FOR_ZERO_CAP_EXCEED = 100.0 # Sıfırdan gelenlerin tavanı aşması için EKSTRA ceza çarpanı
PENALTY_WEIGHT_CONSTRAINTS = 30000

print(f"\n*** AMAÇ FONKSİYONU AĞIRLIKLARI (FİYAT ODAKLI, DÜŞÜK YENİLENEBİLİR HEDEFİ) ***")
print(f"  ALPHA (Fiyat): {ALPHA}, BETA (Emisyon): {BETA}, GAMMA (Yenilenebilir): {GAMMA}")
print(f"*** KISIT: MUTLAK PAY DEĞİŞİMİ (her kaynak için, EN SIKI v3) ***")
print(f"  Maks. Mutlak Değişim: +/-{MAX_ABSOLUTE_CHANGE_PUNITS*100:.1f} puan")
print(f"  Sıfırdan Gelenler İçin Tavan: {ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES*100:.1f} puan (Bu aşılırsa EKSTRA AĞIR CEZA!)")
print(f"  Pay Değişim Cezası (Temel): {PENALTY_WEIGHT_SHARE_CHANGE}")

def create_individual(): # EKSİK FONKSİYON EKLENDİ
    individual = np.random.rand(num_decision_variables)
    individual = np.maximum(individual, 0)
    s = np.sum(individual)
    if s == 0:
        individual = np.full(num_decision_variables, 1/num_decision_variables)
    else:
        individual = individual / s
    return individual.tolist()

def calculate_objectives_and_constraints(individual_shares_list, initial_shares_dict_ref):
    current_shares = dict(zip(share_columns, individual_shares_list))
    y_new = sum(current_shares.get(s,0) for s in renewable_sources_for_y); y_new = np.clip(y_new, 0, 1)
    new_fossil_share_in_current_mix = sum(current_shares.get(s,0) for s in fossil_sources_for_z)
    z_new = (new_fossil_share_in_current_mix / initial_fossil_share_for_z_calc) * z0 if initial_fossil_share_for_z_calc > 1e-9 else (z0 if new_fossil_share_in_current_mix > 1e-9 else 0)

    T0_kWh = T0 * 1_000_000; total_system_cost_euro = 0
    for source_name in current_shares:
        production_kwh_i = current_shares[source_name] * T0_kWh
        op_cost = production_kwh_i * unit_costs_euro_per_kwh[source_name]; total_system_cost_euro += op_cost
        initial_share_for_source = initial_shares_dict_ref.get(source_name, 0)
        increase_in_share = current_shares[source_name] - initial_share_for_source
        if increase_in_share > 0:
            install_cost_val = installation_costs_euro_per_kwh_increase[source_name]
            install_cost_for_source = (increase_in_share * T0_kWh) * install_cost_val
            total_system_cost_euro += install_cost_for_source
    x_new = total_system_cost_euro / T0_kWh if T0_kWh > 1e-9 else x0

    x_norm = x_new / x0 if x0 > 1e-9 else 1.0; z_norm = z_new / z0 if z0 > 1e-9 else 1.0; y_norm = y_new
    cost_constraint_violation = max(0, x_new - (TARGET_PRICE_INCREASE_MAX * x0))
    emission_constraint_violation = max(0, z_new - (TARGET_EMISSION_REDUCTION_MIN * z0))

    share_change_violation = 0
    for s_name in share_columns:
        initial_share = initial_shares_dict_ref.get(s_name, 0)
        current_share = current_shares[s_name]
        
        violation_this_source = 0
        if initial_share < 1e-5: # Başlangıç payı çok düşükse (neredeyse sıfır)
            if current_share > ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES:
                # Bu özel durumu çok daha fazla cezalandır
                violation_this_source = (current_share - ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES) * EXTRA_PENALTY_FACTOR_FOR_ZERO_CAP_EXCEED
            # Bu durumda, normal MAX_ABSOLUTE_CHANGE_PUNITS kontrolüne gerek yok, çünkü zaten 0'dan başlıyor
            # ve tavanı aşıp aşmadığına bakıyoruz.
        else: # Normal payı olan kaynaklar
            absolute_change_val = abs(current_share - initial_share)
            if absolute_change_val > MAX_ABSOLUTE_CHANGE_PUNITS:
                violation_this_source = (absolute_change_val - MAX_ABSOLUTE_CHANGE_PUNITS)
        
        share_change_violation += violation_this_source
                
    return x_norm, y_norm, z_norm, cost_constraint_violation, emission_constraint_violation, share_change_violation, x_new, z_new

def fitness_function(individual_shares_list, initial_shares_dict_ref):
    x_norm, y_norm, z_norm, cost_viol, emission_viol, share_change_viol, _, _ = calculate_objectives_and_constraints(individual_shares_list, initial_shares_dict_ref)
    objective_value = ALPHA * x_norm + BETA * z_norm - GAMMA * y_norm
    penalty = PENALTY_WEIGHT_CONSTRAINTS * (cost_viol + emission_viol) + PENALTY_WEIGHT_SHARE_CHANGE * share_change_viol
    return objective_value + penalty

def selection(population_with_fitness):
    tournament_size = 7; selected_parents = []
    for _ in range(len(population_with_fitness)):
        tournament_contenders = random.sample(population_with_fitness, tournament_size)
        winner = min(tournament_contenders, key=lambda x: x[1]); selected_parents.append(winner[0])
    return selected_parents

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, num_decision_variables - 2) if num_decision_variables > 2 else 1
        child1_s = parent1[:point] + parent2[point:]; child2_s = parent2[:point] + parent1[point:]
        s1 = sum(child1_s); child1 = [c / s1 if s1 > 1e-9 else 1/num_decision_variables for c in child1_s]
        s2 = sum(child2_s); child2 = [c / s2 if s2 > 1e-9 else 1/num_decision_variables for c in child2_s]
        return child1, child2
    return parent1, parent2

def mutate(individual):
    mutated_individual = list(individual)
    num_genes_to_mutate = random.randint(1, max(2, int(0.05 * num_decision_variables))) 
    mutation_strength = 0.010
    indices_to_mutate = random.sample(range(num_decision_variables), num_genes_to_mutate)
    for i in indices_to_mutate:
        change = (random.random() - 0.5) * 2 * mutation_strength
        mutated_individual[i] += change; mutated_individual[i] = max(0, mutated_individual[i])
    s = sum(mutated_individual)
    if s > 1e-9: mutated_individual = [m / s for m in mutated_individual]
    else: mutated_individual = [1/num_decision_variables] * num_decision_variables
    return mutated_individual

population = [create_individual() for _ in range(POPULATION_SIZE)]
best_fitness_overall = float('inf'); best_individual_overall = None; best_stats_overall = {}
print("\nGenetik Algoritma (SIFIR PAY KONTROLÜ VE AĞIR CEZA v2) Başlatılıyor...")
for generation in range(MAX_GENERATIONS):
    population_with_fitness = []
    for ind in population:
        fitness = fitness_function(ind, initial_shares_dict) 
        population_with_fitness.append((ind, fitness))
    current_best_individual, current_best_fitness = min(population_with_fitness, key=lambda x: x[1])
    if current_best_fitness < best_fitness_overall:
        best_fitness_overall = current_best_fitness; best_individual_overall = current_best_individual
        xn, yn, zn, cv, ev, scv, act_x, act_z = calculate_objectives_and_constraints(best_individual_overall, initial_shares_dict)
        best_stats_overall = {"x_norm": xn, "y_norm": yn, "z_norm": zn, "cost_viol": cv, "emiss_viol": ev, 
                              "share_change_viol": scv, "actual_price": act_x, "actual_emission": act_z, "y_percent": yn * 100}
    if (generation + 1) % 25 == 0:
        print(f"J{generation+1}/{MAX_GENERATIONS} - Fit: {best_fitness_overall:.2f} | P: {best_stats_overall.get('actual_price',0):.4f} | R%: {best_stats_overall.get('y_percent',0):.1f} | E: {best_stats_overall.get('actual_emission',0)/1e6:.1f}MT | V(C,E,S): {best_stats_overall.get('cost_viol',0):.1e},{best_stats_overall.get('emiss_viol',0):.1e},{best_stats_overall.get('share_change_viol',0):.1e}")
    parents = selection(population_with_fitness)
    next_population = []; elites = int(0.10 * POPULATION_SIZE)
    sorted_pop = sorted(population_with_fitness, key=lambda x: x[1])
    for i in range(elites): next_population.append(sorted_pop[i][0])
    for i in range(0, POPULATION_SIZE - elites, 2):
        if i + 1 < (POPULATION_SIZE - elites):
            p1,p2 = random.sample(parents,2); c1,c2 = crossover(p1,p2)
            next_population.extend([mutate(c1), mutate(c2)])
        elif len(next_population) < POPULATION_SIZE: next_population.append(mutate(list(random.choice(parents))))
    if len(next_population) < POPULATION_SIZE: next_population.extend([create_individual() for _ in range(POPULATION_SIZE - len(next_population))])
    population = next_population[:POPULATION_SIZE]

print("\nGenetik Algoritma Tamamlandı."); print(f"En İyi Fitness Değeri: {best_fitness_overall:.6f}")
print("\nEn İyi Bulunan Çözüm (Enerji Payları yüzdesel):")
best_shares_dict_final = {}
if best_individual_overall: best_shares_dict_final = dict(zip(share_columns, best_individual_overall))
total_check_sum = 0
print(f"{'Kaynak':<55} {'Başl. (%)':<10} {'Yeni (%)':<10} {'Değ. (puan)':<12} {'Sınır İhlali?':<15}")
print("="*105)
for skey in sorted(share_columns, key=lambda k: initial_shares_dict.get(k,0), reverse=True):
    init_p_val = initial_shares_dict.get(skey,0); final_p_val = best_shares_dict_final.get(skey, 0)
    init_p_pct = init_p_val*100; final_p_pct = final_p_val*100
    abs_change_pu = final_p_pct - init_p_pct; viol_info = ""
    
    limit_this_source = MAX_ABSOLUTE_CHANGE_PUNITS
    if init_p_val < 1e-5: # Başlangıç payı çok düşükse
        if final_p_val > ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES + 1e-6 : # Tavanı aştı mı?
            viol_info = f"Evet (Tavan: {ABSOLUTE_CAP_FOR_NEAR_ZERO_SHARES*100:.1f}p)"
        # Bu durumda bile, eğer tavanı aşmasa da, MAX_ABSOLUTE_CHANGE_PUNITS'i de aşmamalı
        # (çünkü 0'dan başlıyor, değişim=yeni pay)
        elif final_p_val > MAX_ABSOLUTE_CHANGE_PUNITS + 1e-6:
             viol_info = f"Evet (Genel: {MAX_ABSOLUTE_CHANGE_PUNITS*100:.1f}p)"

    elif abs(abs_change_pu/100) > MAX_ABSOLUTE_CHANGE_PUNITS + 1e-6:
        viol_info = "Evet" if abs_change_pu > 0 else "Evet (Azalış)"
            
    if init_p_pct > 0.01 or final_p_pct > 0.01:
        print(f"{skey.replace('_Share_Percent',''):<55} {init_p_pct:<10.2f} {final_p_pct:<10.2f} {abs_change_pu:<12.2f} {viol_info:<15}")
    total_check_sum += final_p_val
print("="*105); print(f"{'TOPLAM':<55} {'':<10} {total_check_sum*100:<10.2f}")

if best_stats_overall:
    print("\nEn İyi Çözümün Özellikleri:")
    print(f"  Fiyat (x_new): {best_stats_overall['actual_price']:.4f} €/kWh (Hedef <= {TARGET_PRICE_INCREASE_MAX*x0:.4f})")
    print(f"  Yenilenebilir Payı (y_new): {best_stats_overall['y_percent']:.2f}%")
    print(f"  CO2 Emisyonu (z_new): {best_stats_overall['actual_emission']:,.0f} T (Hedef <= {TARGET_EMISSION_REDUCTION_MIN*z0:,.0f})")
    print(f"  İhlaller: Maliyet: {best_stats_overall['cost_viol']:.2e}, Emisyon: {best_stats_overall['emiss_viol']:.2e}, Pay Değ.: {best_stats_overall['share_change_viol']:.2e}")
else: print("Uygun çözüm bulunamadı.")
print("\n--- Başlangıç Durumu (2023) ---"); print(f"  Fiyat (x0): {x0:.4f} €/kWh, Yenilenebilir (y0): {y0*100:.2f}%, CO2 (z0): {z0:,.0f} T")

