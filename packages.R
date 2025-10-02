# Install required packages for the analysis
required_packages <- c(
  "tidyverse",
  "dplyr",
  "ggplot2",
  "readr"
)

for(pkg in required_packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg, dependencies = TRUE)
  }
}
