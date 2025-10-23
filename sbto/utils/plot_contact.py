import matplotlib.pyplot as plt
import numpy as np
import sbto.tasks.unitree_g1.g1_constants as const

def plot_contact_achieved_vs_planned(obs_traj, nlp, save_path="contact_achieved_vs_planned.png"):

    cnt_ids = const.G1Sensors.cnt_status_id

    # extract achieved contact status from observations (threshold if continuous)
    achieved_raw = obs_traj[:, cnt_ids]                 # (T, M)
    achieved_bool = (achieved_raw > 0.5).astype(int)    # binary

    # if planner repeated per sensor (common in G1), collapse per foot:
    if achieved_bool.shape[1] % const.cnt_sensor_per_foot == 0:
        Nfeet = achieved_bool.shape[1] // const.cnt_sensor_per_foot
        achieved_per_foot = achieved_bool.reshape(len(achieved_bool), Nfeet, const.cnt_sensor_per_foot).any(axis=2).astype(int)  # (T, Nfeet)
    else:
        achieved_per_foot = achieved_bool  # fallback

    # planned contacts (align lengths)
    planned = nlp.contact_plan
    # if planned was repeated per-sensor, collapse it the same way:
    if planned.shape[-1] % const.cnt_sensor_per_foot == 0 and planned.shape[-1] != achieved_per_foot.shape[1]:
        planned_per_foot = planned.reshape(planned.shape[0], -1, const.cnt_sensor_per_foot).any(axis=2).astype(int)
    else:
        planned_per_foot = planned

    # make sure time dimension matches (trim if needed)
    Tmin = min(achieved_per_foot.shape[0], planned_per_foot.shape[0])
    ach = achieved_per_foot[:Tmin].T   # shape (Nfeet, T)
    pla = planned_per_foot[:Tmin].T     # same

    # plot side-by-side
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].imshow(pla, aspect='auto', origin='lower')
    axs[0].set_title('planned contacts (foot x time)')
    axs[0].set_ylabel('foot'); axs[0].set_xlabel('time step')

    axs[1].imshow(ach, aspect='auto', origin='lower')
    axs[1].set_title('achieved contacts (foot x time)')
    axs[1].set_xlabel('time step')

    plt.tight_layout()
    plt.savefig('contact_achieved_vs_planned.png', dpi=150)
    print('Saved contact_achieved_vs_planned.png')
    plt.close()