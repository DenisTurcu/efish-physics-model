import numpy as np
import scipy.signal as Ssignal
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from Worm import Worm


class SmallSphericalWorm(Worm):
    ''' Represents a small, spherical worm of given radius. Parent class "Worm": '''
    __doc__ += Worm.__doc__

    def __init__(self, radius=0.01, resistance=None, capacitance=None, derivative_filter=[1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280], _init_tests=True, **kwds):
        super().__init__(**kwds, _init_tests=False)
        self.radius = convert2mainSI(radius)
        self.der_filt = derivative_filter
        self.initiate_electrical_properties(resistance, capacitance)
        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

        
    def differential_equation_coefficients(self, sig_water, eps_r_water):
        ''' Compute the coefficients of the differential equaiton to solve (Eq. (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf). '''
        Rw = 2 / (np.pi * self.radius * sig_water)
        Cw = np.pi * self.radius * self.eps0 * eps_r_water / 2
        f0 = 1/self.R - 1/Rw
        f1 = self.C - Cw
        p0 = 1/self.R + 2/Rw
        p1 = self.C + 2 * Cw
        return f0, f1, p0, p1


    def dipole_waveform_single_fish(self, wave_form, sig_water, eps_r_water, sampling_frequency):
        ''' Compute the wave form of the induced dipole (different than the waveform of the EOD)
            Solve Eq. (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf
                p0 * phi(t) + p1 * (d phi)/dt = f0 * f(t) + f1 (d f)/dt '''
        assert wave_form.shape == wave_form.reshape(-1).shape, 'Wave form should be shape (T,) - i.e. not (T,1) or (1,T).'

        wave_form_der = np.convolve(self.der_filt[::-1], wave_form, 'same') * sampling_frequency
        f0, f1, p0, p1 = self.differential_equation_coefficients(sig_water, eps_r_water)
        dipole_wave_form = np.zeros(wave_form.shape[0])
        old_settings = np.seterr(over='ignore', invalid='ignore')
        for j in range(1, wave_form.shape[0]):
            dipole_wave_form[j] = dipole_wave_form[j-1] + (f0 * wave_form[j-1] + f1 * wave_form_der[j-1] - p0 * dipole_wave_form[j-1]) / (p1 * sampling_frequency)
        np.seterr(**old_settings)
        # Ensure proper solution for very small tau=RC
        dipole_wave_form = self.dipole_waveform_quality_check(dipole_wave_form, wave_form, wave_form_der, sampling_frequency, f0, f1, p0, p1)
        return dipole_wave_form
    

    def dipole_waveform_quality_check(self, dipole_wave_form, wave_form, wave_form_der, sampling_frequency, f0, f1, p0, p1, resolution_multiplication=5, runtime_err_factor=5000, check_err=1.1e-2):
        resolution_factor = 1
        discrepancy_size = np.abs(p0 * dipole_wave_form + p1 * np.convolve(self.der_filt[::-1], dipole_wave_form, 'same') * sampling_frequency - f0 * wave_form - f1 * wave_form_der).max()
        while (discrepancy_size > check_err) or np.isnan(discrepancy_size):
            # # print message to inform of poor estimate of the dipole waveform
            # print(f'!!! Dipole waveform quality check has failed on resistance {self.R} and capacitance {self.C} at super resolution {resolution_factor}. Discrepancy size was {discrepancy_size}.', end=' ')
            resolution_factor *= resolution_multiplication
            if resolution_factor > runtime_err_factor:
                # raise RuntimeError(f'Could not solve differential equation (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf with small enough error. Failed due to resistance {self.R} and capacitance {self.C} with maximum resolution factor of {resolution_factor/resolution_multiplication}. Discrepancy size was {discrepancy_size}.')
                print(f'____________ Could not solve differential equation (7) in "Electric_Fish_Numerically_Stable_Solution" Overleaf with small enough error. Failed due to radius {self.radius}, resistance {self.R} and capacitance {self.C} with maximum resolution factor of {resolution_factor/resolution_multiplication}. Discrepancy size was {discrepancy_size}. Returning best estimated "dipole_wave_form".')
                return dipole_wave_form  # np.zeros(wave_form.shape[0])
            # print(f'Trying again at super resolution {resolution_factor}.')
            wave_form_resampled = Ssignal.resample(wave_form, wave_form.shape[0]*resolution_factor)
            wave_form_der_resampled = np.convolve(self.der_filt[::-1], wave_form_resampled, 'same') * sampling_frequency * resolution_factor
            dipole_wave_form = np.zeros(wave_form_resampled.shape[0])
            old_settings = np.seterr(over='ignore', invalid='ignore')
            for j in range(1, wave_form_resampled.shape[0]):
                dipole_wave_form[j] = dipole_wave_form[j-1] + (f0 * wave_form_resampled[j-1] + f1 * wave_form_der_resampled[j-1] - p0 * dipole_wave_form[j-1]) / (p1 * sampling_frequency * resolution_factor)
            np.seterr(**old_settings)
            dipole_wave_form = dipole_wave_form[::resolution_factor]
            discrepancy_size = np.abs(p0 * dipole_wave_form + p1 * np.convolve(self.der_filt[::-1], dipole_wave_form, 'same') * sampling_frequency - f0 * wave_form - f1 * wave_form_der).max()
        # print(f'VVV Dipole waveform quality check has succeeded on resistance {self.R} and capacitance {self.C} at super resolution {resolution_factor}. Discrepancy size was {discrepancy_size}.\n')
        return dipole_wave_form


    def perturbation_magnitude_single_fish(self, receptor_locs, E_field, return_potential=True, return_field=True, include_radius_factor=True):
        ''' Returns the electric potential/field perturbation magnitude, i.e. without the temporal trace, at given locations. '''
        E_field   = E_field.reshape(3)
        assert E_field.shape == np.zeros(3).shape, 'Electric field at the center of the Worm should be shape (3,) - i.e. not (3,1) or (1,3).'
        
        r_relative = receptor_locs - self.r_vec
        assert r_relative.shape == receptor_locs.shape, 'The relative distance should match the shape of all fish receptor locations.'
        assert (r_relative[0,:] == (receptor_locs[0,:]-self.r_vec)).all(), 'Individual examples of relative distance should match values.'

        radius_factor = 1 if not include_radius_factor else np.power(self.radius, 3)

        potential_perturbation = None
        E_perturbation = None
        
        if return_potential:
            # compute electric potential perturbation at given locations according to EM computations
            potential_perturbation = radius_factor * (E_field * r_relative).sum(1) / np.power(np.linalg.norm(r_relative, axis=1), 3)
            assert (potential_perturbation.shape == receptor_locs[:,0].shape), 'Electric potential should be shape (N,) at this stage'

        if return_field:
            # compute electric field perturbation at given locations according to EM computations
            E_perturbation = 3 * (E_field * r_relative).sum(1, keepdims=True) * r_relative - np.power(np.linalg.norm(r_relative, axis=1, keepdims=True), 2) * E_field
            E_perturbation *= radius_factor / np.power(np.linalg.norm(r_relative, axis=1, keepdims=True), 5)
            assert E_perturbation.shape == receptor_locs.shape, 'The E field perturbation should match the shape of all fish points.'
        
        return potential_perturbation, E_perturbation


    def perturbation_trace(self, receptor_locs, aquarium_obj, return_potential=True, return_field=True):
        ''' Returns electric potential/field perturbations computed at given locations. '''        
        potential_perturbation_total = 0
        E_perturbation_total = 0
        for i, fish in enumerate(aquarium_obj.get_fish()):
            # compute electric field at self.r_vec
            _, E_field, wave_form = aquarium_obj.electric_potential_and_field_single_fish(self.r_vec[np.newaxis], fish_id=i,
                                                                                          return_potential=False,
                                                                                          return_field=True)

            potential_perturbation, E_perturbation = self.perturbation_magnitude_single_fish(receptor_locs=receptor_locs, 
                                                                                             E_field=E_field, 
                                                                                             return_potential=return_potential, 
                                                                                             return_field=return_field, 
                                                                                             include_radius_factor=True)

            dipole_wave_form = self.dipole_waveform_single_fish(wave_form=wave_form, 
                                                                sig_water=aquarium_obj.get_conductivity(), 
                                                                eps_r_water=aquarium_obj.get_relative_permitivity(),
                                                                sampling_frequency=fish.get_sampling_rate())
            
            if return_potential:
                potential_perturbation = potential_perturbation[:,np.newaxis] * dipole_wave_form
                assert potential_perturbation.shape[0] == receptor_locs.shape[0], 'Electric potential should be shape (N,T) at this stage'
                assert potential_perturbation.shape[1] == wave_form.shape[0], 'Electric potential should be shape (N,T) at this stage'
                potential_perturbation_total += potential_perturbation

            if return_field:
                E_perturbation = E_perturbation[:,:,np.newaxis] * dipole_wave_form
                assert (E_perturbation.shape == np.array([receptor_locs.shape[0],3,wave_form.shape[0]])).all(), 'The E field perturbation should be shape (N x 3 x T).'
                E_perturbation_total += E_perturbation

        return potential_perturbation_total, E_perturbation_total


    def initiate_electrical_properties(self, resistance, capacitance):
        if resistance is None:
            print('!!! Using the worm material conductivity to estimate its resistance.')
            self.R = 2 / (np.pi * self.radius * self.sig)
        else:
            self.R = convert2mainSI(resistance)
        if capacitance is None:
            print('!!! Using the worm material relative permitivity to estimate its capacitance.')
            self.C = np.pi * self.radius * self.eps0 * self.eps_r / 2
        else:
            self.C = convert2mainSI(capacitance)
        pass


    def graph_obj(self, size_scale=10, units_prefix=''):
        ''' Create a plotly graphical object to insert into a plot at a later time. '''
        graph_obj = go.Scatter3d(x=[add_prefix(self.r_vec[0], units_prefix)], 
                                 y=[add_prefix(self.r_vec[1], units_prefix)], 
                                 z=[add_prefix(self.r_vec[2], units_prefix)],
                                 showlegend=False, mode='markers', 
                                 marker=dict(size=self.radius*size_scale,
                                             color='pink', opacity=0.9),
                                 )
        return graph_obj


    def is_equal(self, other, simple_return=True):
        ''' Compare current worm to another worm for equality. '''
        _, thruth_values, comparison  = super().is_equal(other, simple_return=False)
        if isinstance(other, self.__class__):
            thruth_values.append(np.abs(self.radius - other.get_radius()) < np.min([self.assert_err, other.assert_err]))
            comparison.append('radius')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value  
        else:
            return thruth_value, thruth_values, comparison


    def details(self):
        ''' Provides the details of the object. '''
        details_string =  super().details()
        details_string += f'This worm type is small and spherical, with radius {self.radius}.\n'
        return details_string


    def get_radius(self):
        ''' Extract the worm's position [x, y, z]. '''
        return self.radius


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['radius', 'resistance', 'capacitance']
        return inp_args


    def run_tests(self):
        super().run_tests()
        return 'Success!'















