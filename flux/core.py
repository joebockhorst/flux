from prettytable import PrettyTable
from dataclasses import dataclass, asdict, astuple
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt

def end(msg):
    import sys
    print(msg)
    sys.exit(1)

class FluxData:

    def __init__(self, filename):
        self.df = pd.read_csv(filename)

        for c in ['GroupLevel1_Pretreatment', 'GroupLevel2_Treatment', 'HRP_ngml']:
            if c not in self.df.columns:
                #print(f"Error. Column missing: {c!r}")
                end(f"Error. Column missing: {c!r}")

    @property
    def pretreatments(self):
        results = []
        for pt in self.df.GroupLevel1_Pretreatment:
            if pt not in results:
                results.append(pt)
        return results

    @property
    def treatments(self):
        results = []
        for tr in self.df.GroupLevel2_Treatment:
            if tr not in results:
                results.append(tr)
        return results

    def get_flux(self, pretreatment:str, treatment:str):
        idx = list(a and b for a, b in zip(self.df.GroupLevel1_Pretreatment == pretreatment, self.df.GroupLevel2_Treatment == treatment))
        tmp = self.df.HRP_ngml[idx]
        result = pd.Series([val for val in tmp if not pd.isna(val)])
        if len(result) < len(tmp):
            print(f"warning: NA flux values for {pretreatment}-{treatment}")
        if len(result) == 0:
            raise Exception(f"No flux values for {pretreatment}-{treatment}")
        return result


@dataclass
class IntraGroupData:
    pre : str
    tr : str
    mean_flux : float
    SEM : float
    perc_change : float
    pvalue : float

    def table_row(self):
        return self.pre, self.tr, f"{self.mean_flux:6.2f}", f"{self.SEM:6.2f}", f"{self.perc_change:6.0f}", f"{self.pvalue:7.4f}"

class IntraGroup:

    def __init__(self, fluxdata, reference_treatment):
        self.fluxdata = fluxdata
        self.reference_treatment = reference_treatment

    def get_one_intra_group_data(self, pre:str, tr:str):
        flux = self.fluxdata.get_flux(pre, tr)
        flux_ref = self.fluxdata.get_flux(pre, self.reference_treatment)
        t_out = ttest_ind(flux, flux_ref, equal_var=False, alternative="two-sided")
        p = t_out.pvalue
        perc_change = 100.0 * (flux.mean() - flux_ref.mean()) / flux_ref.mean()
        sem = np.std(flux) / np.sqrt(len(flux))
        igd = IntraGroupData(pre=pre, tr=tr, mean_flux=flux.mean(), SEM=sem, perc_change=perc_change, pvalue=p)
        return igd

    def get_intra_group_data(self) -> list[IntraGroupData]:
        igds = []
        for pre, tr in itertools.product(self.fluxdata.pretreatments, self.fluxdata.treatments):
            igds.append(self.get_one_intra_group_data(pre, tr))
        return igds

    def get_table(self):
        table = PrettyTable()
        title = f"Intra-group change relative to {self.reference_treatment}"

        field2attr = {
            "Pretreatment" : "pre",
            "Treatment" : "tr",
            "Mean" : "mean_flux",
            "SEM" : "SEM",
            "% change" : "perc_change",
            "p" : "pvalue"
        }

        table.title = title
        table.field_names = list(field2attr.keys())
        for igd in self.get_intra_group_data():
            table.add_row(igd.table_row())

        table.align["Pretreatment"] = "l"
        table.align["Treatment"] = "l"
        return table

@dataclass
class InterGroupData:
    pre : str
    tr : str
    coeff : float
    SE : float
    p : float

    def table_row(self):
        return self.pre, self.tr, f"{self.coeff:6.2f}", f"{self.SE:6.2f}", f"{self.p:7.4f}"


class InterGroup:

    def __init__(self, fluxdata, ref_pretreatment, ref_treatment):
        self.fluxdata = fluxdata
        self.ref_pretreatment = ref_pretreatment
        self.ref_treatment = ref_treatment

        if ref_pretreatment not in self.fluxdata.pretreatments:
            end(f"Error: Pretreatment not found {ref_pretreatment!r}")

        if ref_treatment not in self.fluxdata.treatments:
            end(f"Error: Treatment not found {ref_treatment!r}")

    def get_xs_interaction(self, pt, tr):
        index = ["Tr", "Pre", "PreAndTr"]
        val = [
            1 if tr != self.ref_treatment else 0,
            1 if pt != self.ref_pretreatment else 0,
            1 if tr != self.ref_treatment and pt != self.ref_pretreatment else 0
        ]
        return pd.Series(val, index=index)

    def interaction_model_results(self, pt, tr):
        assert self.ref_pretreatment != pt
        assert self.ref_treatment != tr
        fluxes = {}
        for pt_val in [self.ref_pretreatment, pt]:
            for tr_val in [self.ref_treatment, tr]:
                fluxes[pt_val, tr_val] = self.fluxdata.get_flux(pt_val, tr_val)

        n = sum(len(fl) for fl in fluxes.values())
        varnames = [pt, tr, "Interaction"]
        n_vars = len(varnames)

        X = np.zeros((n, n_vars)) + np.nan
        y = np.zeros((n,)) + np.nan
        nrows = 0
        for pt_val in [self.ref_pretreatment, pt]:
            for tr_val in [self.ref_treatment, tr]:
                flux = fluxes[pt_val, tr_val]
                xs = self.get_xs_interaction(pt_val, tr_val)
                X[nrows:nrows + len(flux), :] = xs
                y[nrows:nrows + len(flux)] = flux
                nrows += len(flux)

        assert np.isnan(X).sum() == 0
        assert np.isnan(y).sum() == 0
        X = sm.add_constant(X)

        def norm(c):
            c = c.replace(" ", "_")
            c = c.replace("1", "_one_")
            c = c.replace("%", "_percent_")
            c = c.replace("0", "_zero_")
            c = c.replace(".", "_dot_")
            c = c.replace("/", "_per_")
            c = c.replace(",", "_")
            c = c.replace("-", "_dash_")
            return c

        cols = [norm(col) for col in varnames]

        data = pd.DataFrame(X[:, 1:], columns=cols)
        fmla = f"flux ~ {' + '.join(data.columns)}"
        data["flux"] = y
        mod = smf.ols(formula=fmla, data=data)
        result = mod.fit()
        return result

    def get_inter_group_data(self) -> list[InterGroupData]:
        result = []
        for pt in (x for x in self.fluxdata.pretreatments if x !=self.ref_pretreatment):
            for tr in (x for x in self.fluxdata.treatments if x != self.ref_treatment):
                results = self.interaction_model_results(pt=pt, tr=tr)
                coeff = results.params.Interaction
                stderr = results.bse.Interaction
                p = results.pvalues.Interaction

                igd = InterGroupData(pre=pt, tr=tr, coeff=coeff, SE=stderr, p=p)
                result.append(igd)
        return result


    def get_table(self):
        table = PrettyTable()
        title = f"Inter-group change relative to {self.ref_pretreatment!r}"

        pre: str
        tr: str
        coeff: float
        SE: float
        p: float

        field2attr = {
            "Pretreatment" : "pre",
            "Treatment" : "tr",
            "Coeff" : "coeff",
            "SE" : "SE",
            "p" : "p"
        }

        table.title = title
        table.field_names = list(field2attr.keys())
        for igd in self.get_inter_group_data():
            table.add_row(igd.table_row())

        table.align["Pretreatment"] = "l"
        table.align["Treatment"] = "l"
        return table


def intragroup_vs_control_table(filename):
    fluxdata = FluxData(filename)
    intra_gd = IntraGroup(fluxdata, reference_treatment='Control pH7.4')
    print(intra_gd.get_table())


def intergroup_vs_control_table(filename, ref_pretreatment):
    fluxdata = FluxData(filename)
    inter_gd = InterGroup(fluxdata=fluxdata, ref_pretreatment=ref_pretreatment, ref_treatment='Control pH7.4')
    print(inter_gd.get_table())



#
# CONTROL = 'Control pH7.4'
# NOPRETREATMENT = 'No Pretreatment'
# GAV_DUAL = 'Gaviscon Dual Action'
# GAV_ADV = 'Gaviscon Advanced'
# PLACEBO = "Placebo"
#
# PH5_1MGPERML = '1mg/ml pepsin, pH5'
# ONEMGPH5 = '1mg/ml pepsin, pH5'
# ONEMGPH4 = '1mg/ml pepsin, pH4'
# POINT1MGPERMLPH3 = "0.1mg/ml pepsin, pH3"
#
#
#
#
# treatments = [ 'Control pH7.4', '1mg/ml pepsin, pH5', '1mg/ml pepsin, pH4', '0.1mg/ml pepsin, pH3', 'pH3', '0.1% Tx-100 ', ]
# pretreatments = ['No Pretreatment',  'Placebo', 'Gaviscon Advanced', 'Gaviscon Dual Action']
#
# def get_flux(df, pt, tr):
#     idx = list(a and b for a, b in zip(df.GroupLevel1_Pretreatment == pt, df.GroupLevel2_Treatment == tr))
#     return pd.Series(df.HRP_ngml[idx])
#


# def table_row(results):
#     coeff = results.params.Interaction
#     stderr = results.bse.Interaction
#     p = results.pvalues.Interaction
#
#     return coeff, stderr, p
#
#
# pt_h = "Pretreatment"
# tr_h = "Treatment"
# coeff_h = "Coeff"
# se_h = "SE"
# p_h = "p"
#
# ################ Inter-group change relative to No Pretreatment ############
# trs = [ONEMGPH5, ONEMGPH4, POINT1MGPERMLPH3]
# pts = [PLACEBO, GAV_ADV, GAV_DUAL]
# baseline_tr = NOPRETREATMENT
# header = f"{pt_h:^24} {tr_h:^24} {coeff_h:^7} {se_h:^6} {p_h:^7}"
#
# print("               Inter-group change relative to No Pretreatment")
# print(header)
# print(f"-"*70)
#
# for pt in pts:
#     for tr in trs:
#         lines, results = do_model(pt=pt, tr=tr, interaction=True, baseline_pt=baseline_tr)
#         coeff, se, p = table_row(results=results)
#         print(f"{pt:24} {tr:24} {coeff:6.2f} {se:6.2f} {p:7.4f}")


#
# ################ Inter-group change relative to Placebo ############
# trs = [ONEMGPH5, ONEMGPH4, POINT1MGPERMLPH3]
# baseline_tr = PLACEBO
# print("               Inter-group change relative to Placebo")
# print(header)
# print(f"-"*70)
#
# pt = GAV_ADV
# for tr in trs:
#     lines, results = do_model(pt=pt, tr=tr, interaction=True, baseline_pt=baseline_tr)
#     coeff, se, p = table_row(results=results)
#     print(f"{pt:24} {tr:24} {coeff:5.2f} {se:5.2f} {p:7.4f}")
# pt = GAV_DUAL
# for tr in trs:
#     lines, results = do_model(pt=pt, tr=tr, interaction=True, baseline_pt=baseline_tr)
#     coeff, se, p = table_row(results=results)
#     print(f"{pt:24} {tr:24} {coeff:5.2f} {se:5.2f} {p:7.4f}")
#
#
# ################ Inter-group change relative to Gaviscon Advanced ############
# trs = [ONEMGPH5, ONEMGPH4, POINT1MGPERMLPH3]
# pts = [GAV_DUAL]
# baseline_tr = GAV_ADV
# print("               Inter-group change relative to Gaviscon Advanced")
# print(header)
# print(f"-"*70)
#
# for pt in pts:
#     for tr in trs:
#         lines, results = do_model(pt=pt, tr=tr, interaction=True, baseline_pt=baseline_tr)
#         coeff, se, p = table_row(results=results)
#         print(f"{pt:24} {tr:24} {coeff:6.2f} {se:6.2f} {p:7.4f}")
