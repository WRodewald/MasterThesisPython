import numpy as np

# helper functions to work with the LF Rd glottal flow model model


# function calculates LF parameters from 1-parameter LF-Rd
def convert_Rd_to_tx(Rd):
	Rap = (-1.  +  4.8 * Rd) * 0.01
	Rkp = (22.4 + 11.8 * Rd) * 0.01
	RgpInv = (0.44*Rd - 2.*Rap - 4.8*Rap*Rkp) / (1.2*Rkp*Rkp + 0.5*Rkp) 

	tp = 0.5 * RgpInv
	te = tp * (Rkp+1)
	ta = Rap

	return tp, te, ta

# function approximates alpha from Rd with a 2nd/1st order rational polynom
def approximate_alpha(Rd):
	p1 = -0.09818
	p2 =  -0.6536
	p3 =    3.689
	q1 =  0.04759

	return (p1 * Rd * Rd + p2 * Rd + p3) / (Rd + q1)

# function approximates epsilon from Rd with a 2nd/1st order rational polynom
def approximate_epsilon(Rd):
	p1 =  -1.671
	p2 =   2.813
	p3 =   19.71
	q1 = -0.2104

	return (p1 * Rd * Rd + p2 * Rd + p3) / (Rd + q1)


def calculate_waveform(Rd, t, align_te = False):

	a = approximate_alpha(Rd)
	e = approximate_epsilon(Rd)
	tp, te, ta = convert_Rd_to_tx(Rd)

	tx = t
	if(align_te):
		tx += te
		tx -= np.floor(tx)

	

	# calculate E0 with minor correction step to compensate for alpha / epsilon approximation
	E0 = -1. / (np.exp(a*te) * np.sin(te * np.pi / tp ))
	E0 *= -(-1. / (e * ta)) * (np.exp(-e * (te-te)) - np.exp(-e * (1-te)))

	y1 =  E0 * np.exp(a * tx) * np.sin(tx * np.pi / tp)

	y2 = (-1. / (e * ta)) * (np.exp(-e * (tx-te)) - np.exp(-e * (1-te)))

	return y1 * (tx <= te) + y2 * (tx > te)