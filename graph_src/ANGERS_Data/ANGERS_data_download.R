library(prospect)

################## Download ANGERS data set 
LeafDB <- download_LeafDB(dbName = "ANGERS")

# Extract reflectance data and transpose
reflectance_data <- as.data.frame(t(LeafDB$Refl))
colnames(reflectance_data) <- LeafDB$lambda  # spectral column names

# Extract the 6 metadata / biochemical variables
metadata <- LeafDB$DataBioch  # 6 columns

# Add ID column
df <- cbind(ID = seq_len(nrow(reflectance_data)), metadata, reflectance_data)

setwd("...") # Replace with your actual path
write.csv(df, "ANGERS_spectral_data.csv", row.names = FALSE)
