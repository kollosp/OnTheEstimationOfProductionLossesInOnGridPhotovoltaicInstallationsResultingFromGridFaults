if __name__ == "__main__": import __config__
import os
from Analysis import Analysis
from Analysis import TEXTS
from Simulation import Simulation

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
    prone_installation_id = 3
    healthy_installation_id = 0
    limit_voltage = 254

    model_factory_1 = Analysis.get_nlastperiods_model
    model_factory_2 = Analysis.get_seaippf_model
    seaippf_params = {
        "Bins":23.000000,
        "conv2D_shape_factor":0.042373,
        "regressor_degrees":14.000000,
        "bandwidth":0.017857,
        "iterations": 2,
        "iterative_regressor_degrees":11.000000,
        "stretch": False
    }
    model_params = seaippf_params
    model_factory = model_factory_2

    a = Analysis()

    txt = ""
    txt += a.prone_healthy_classification(installation_ids=list(range(0,4)), limit_voltage=limit_voltage)

    figures = [
        a.plot_power(),
        a.plot_voltage(installation_id=prone_installation_id, title=TEXTS["prone_installation_OV_vs_power"],
                       limit_voltage=limit_voltage),  # prone installation
        a.plot_voltage(installation_id=healthy_installation_id, title=TEXTS["healthy_installation_OV_vs_power"],
                       limit_voltage=limit_voltage),  # healthy installation
        a.plot_potential_losses(prone_installation_id=prone_installation_id,
                                healthy_installation_id=healthy_installation_id, limit_voltage=limit_voltage),
        a.plot_model_vs_power(model_factory=model_factory, prone_installation_id=prone_installation_id,
                              healthy_installation_id=healthy_installation_id, model_parameters=model_params),
        a.plot_model_vs_looses(model_factory=model_factory, prone_installation_id=prone_installation_id,
                               healthy_installation_id=healthy_installation_id, limit_voltage=limit_voltage,
                               model_parameters=model_params),
        a.plot_model_vs_looses_metric(model_factory=model_factory, prone_installation_id=prone_installation_id,
                                      healthy_installation_id=healthy_installation_id, limit_voltage=limit_voltage,
                                      model_parameters=model_params),
        a.plot_seaippf_model_parameters(prone_installation_id=prone_installation_id,
                                        healthy_installation_id=healthy_installation_id, config=0, model_parameters=model_params),
        a.plot_model_test_dataset(prone_installation_id=prone_installation_id,
                                  healthy_installation_id=healthy_installation_id, limit_voltage=limit_voltage)
    ]

    show_or_save = False #set False to display images or True to save them in `cm` directory
    if show_or_save:
        DIR = "cm"
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        pp = PdfPages(os.path.join(DIR, 'paper_images.pdf'))
        for f in figures:
            print(f"Working in {str(f)}...")
            if isinstance(f, list):
                for ff in f:
                    pp.savefig(ff)
            else:
                pp.savefig(f)
        pp.close()
    else:
        for f in figures:
            print(f"Working in {str(f)}...")
            if isinstance(f, list):
                for ff in f:
                    ff.show()
            else:
                f.show()
        plt.show()
