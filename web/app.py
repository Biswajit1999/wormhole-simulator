"""Interactive Streamlit dashboard for wormhole-sim.

Run from the repository root:

    streamlit run web/app.py

Lets a user pick a metric and parameters and see, live: the embedding profile,
the null-energy-condition profile across the throat, and the gravitational-
lensing deflection curve.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("Install streamlit:  pip install streamlit, then `streamlit run web/app.py`")

import matplotlib.pyplot as plt

from core.metrics import MorrisThorne, EllisBronnikov, ChargedWormhole
from core import stress_energy as se
from visualization.embed import embedding_profile
from visualization.raytrace import deflection_angle

st.set_page_config(page_title="wormhole-sim", layout="wide")
st.title("wormhole-sim · interactive explorer")

with st.sidebar:
    st.header("Metric")
    kind = st.selectbox("Family", ["Morris-Thorne", "Ellis-Bronnikov", "Charged"])
    b0 = st.slider("throat radius b0", 0.4, 3.0, 1.0, 0.05)
    Q = st.slider("charge Q (charged only)", 0.0, 0.9, 0.3, 0.05)

if kind == "Morris-Thorne":
    wh = MorrisThorne(b0=b0)
    shape = wh.shape
elif kind == "Ellis-Bronnikov":
    wh = EllisBronnikov(b0=b0)
    shape = lambda r: b0 ** 2 / r
else:
    wh = ChargedWormhole(b0=b0, Q=Q)
    shape = wh.shape

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Embedding profile z(r)")
    r, z = embedding_profile(shape, b0, r_max=5 * b0)
    fig, ax = plt.subplots()
    ax.plot(r, z, "c"); ax.plot(r, -z, "c")
    ax.set_xlabel("r"); ax.set_ylabel("z"); ax.set_title("throat cross-section")
    st.pyplot(fig)

with c2:
    st.subheader("Null energy condition")
    rs = np.linspace(b0 * 1.02, 4 * b0, 50)
    nec = [se.null_energy_condition(wh, np.array([0.0, ri, np.pi / 2, 0.0]), n_dirs=16) for ri in rs]
    fig, ax = plt.subplots()
    ax.axhline(0, color="k", lw=0.6); ax.plot(rs, nec, "r")
    ax.fill_between(rs, nec, 0, where=np.array(nec) < 0, color="red", alpha=0.2)
    ax.set_xlabel("r"); ax.set_ylabel("min T_kk"); ax.set_title("NEC (negative = exotic)")
    st.pyplot(fig)

with c3:
    st.subheader("Lensing deflection")
    rhos = np.linspace(b0 * 1.05, 8.0, 60)
    ang = [np.degrees(deflection_angle(b0, rho)) for rho in rhos]
    fig, ax = plt.subplots()
    ax.plot(rhos, ang, "g")
    ax.set_xlabel("impact parameter"); ax.set_ylabel("deflection (deg)")
    ax.set_title("Ellis-type bending")
    st.pyplot(fig)

st.caption("All curves are computed live from the same core.metrics objects.")
