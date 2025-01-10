import data_utils as du

########
# CWRU #
########

print("CWRU")
print("----")

# Load data
CWRU_data = du.get_CWRU_data()

# To iterate through all data
print("Dataframe head:")
print(CWRU_data.head(3))
print(f"Dataframe shape: {CWRU_data.shape}")
print()

for i, row in CWRU_data.iterrows():
    print(
        f"{row['measurement location']}, {row['fault location']}, {row['fault diameter']} mils, {row['fault orientation']}, {row['sampling rate']} kHz, {row['motor load']} HP"
    )
    print(f"Measurement shape: {row['measurement'].shape}")
    # break  # ! Remove this to go through all rows

print()
print()
# To get one sample
sample, rpm, tags = du.get_CWRU_measurement("DE", "DE", "ir", 7, "-", 12, 1)
print(f"Sample shape: {sample.shape}")
print(f"Sample rpm: {rpm}")
print(f"Sample tags: {tags}")


#############
# Paderborn #
#############

print()
print()
print("Paderborn")
print("---------")
print()

# Load data
PU_data = du.get_PU_data()

# To iterate through all data
print("Dataframe head:")
print(PU_data.head(3))
print(f"Dataframe shape: {PU_data.shape}")
print()

for i, row in PU_data.iterrows():
    print(
        f"{row['bearing']}, {row['sample num']}, {row['fault type']}, {row['rpm']} mils, {row['motor load']}, {row['radial force']}"
    )
    print(f"Measurement shape: {row['measurement'].shape}")
    break  # ! Remove this to go through all rows

print()
print()
# To get one sample
sample, fault_type = du.get_PU_measurement("KA01", 1, 1500, 7, 1000, PU_data=PU_data)
print(f"Sample shape: {sample.shape}")
print(f"Sample fault type: {fault_type}")
