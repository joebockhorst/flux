from flux.core import intragroup_vs_control_table, intergroup_vs_control_table

if __name__ == "__main__":
    filename = "data/092121_HRPFlux_BART_read at 20m.csv"
    filename = "data/100521_HRPFlux_TVCHPV_5m.csv"
    filename = "data/100621_HRPFlux_TVCHPV_10m.csv"
    filename = "data/100621_HRPFlux_TVCHPV_5m.csv"
    intragroup_vs_control_table(filename)

    intergroup_vs_control_table(filename, "No Pretreatment")
    intergroup_vs_control_table(filename, "Placebo")
    intergroup_vs_control_table(filename, "Gaviscon Advanced")
