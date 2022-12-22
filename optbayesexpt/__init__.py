from optbayesexpt.constants import version as __version__
from optbayesexpt.constants import GOT_NUMBA
from optbayesexpt.particlepdf import ParticlePDF
from optbayesexpt.obe_base import OptBayesExpt
from optbayesexpt.obe_noiseparam import OptBayesExptNoiseParameter
from optbayesexpt.obe_socket import Socket
from optbayesexpt.obe_server import OBE_Server
from optbayesexpt.obe_utils import MeasurementSimulator, trace_sort

try:
    # Try loading differential_entropy from scipy.stats
    from scipy.stats import differential_entropy
except ImportError:
    # If there are problems, use the version included in obe_utils.py
    from optbayesexpt.obe_utils import differential_entropy

