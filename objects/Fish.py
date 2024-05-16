import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import time

import sys
sys.path.append('../helper_functions/')

from prefixes import add_prefix, remove_prefix, return_numerical_prefix, parse_prefix
from conversions import rho24sig, convert2mainSI
from ElectricObject import ElectricObject


class Fish(ElectricObject):
    ''' Represents a fish. The eod waveform is always normalized to have maximum at 1 - the magnitude of the EOD can be adjusted via the strength of the point currents magnitudes Parent class "ElectricObject": '''
    __doc__ += ElectricObject.__doc__

    def __init__(self, receptors_locations=np.array([]).reshape(0,3),
                       receptors_normals=np.array([]).reshape(0,3),
                       point_currents_magnitudes=np.array([]).reshape(0),
                       point_currents_locations=np.array([]).reshape(0,3),
                       receptor_filters=np.array([]).reshape(0,1000),
                       eod_waveform=np.arange(1000),
                       skin_resistivity=1,
                       sampling_rate=(2.5, 'M'),  # MHz
                       eod_delay=(0,''), # s
                       _init_tests=True, **kwds):
        super().__init__(**kwds, _init_tests=False)

        # set an ID for this fish
        self.ID = int(time.time()*1000)

        # initialize the Fish receptors and point currents
        self.receptors_locations = receptors_locations
        self.receptors_normals   = receptors_normals
        self.point_currents_magnitudes = point_currents_magnitudes
        self.point_currents_locations =  point_currents_locations

        # EOD related attributes
        self.sampling_rate = convert2mainSI(sampling_rate)
        self.eod_waveform  = eod_waveform
        self.eod_length    = eod_waveform.shape[0]
        self.update_eod_waveform_and_delay(eod_waveform, sampling_rate, eod_delay)
        
        # electric related attributes
        self.skin_rho = convert2mainSI(skin_resistivity)

        # innitialize the convolutional filters
        self.receptor_filters = receptor_filters

        # initialization methods
        self._initialize_input_argument_names()

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())


    def compute_transdermal_signal(self, E_field, water_conductivity, temporal_wave_form=None):
        ''' Computes the electric image perturbation at the skin of the fish. '''
        assert (E_field.shape[0] == self.receptors_locations.shape[0]) and (E_field.shape[1] == 3), 'Electric field input should be shape (N x 3) or (N x 3 x T).'

        if len(E_field.shape) == 2:
            if temporal_wave_form is None:
                raise ValueError('The temporal wave form of the eods needs to be reflected in the 3rd dimension of the field or be provided separately.')
            # compute image perturbation, proportional to the electric field normal to the fish's skin
            transdermal_signal  = (self.receptors_normals * E_field).sum(1, keepdims=True) * self.skin_rho * water_conductivity
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (transdermal_signal.shape[1] == 1), 'Electric image perturbation shape should be the number of fish receptors by 1.'
            # include temporal component in the image perturbation
            transdermal_signal = transdermal_signal * temporal_wave_form
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (transdermal_signal.shape[1] == temporal_wave_form.shape[0]), 'Electric image perturbation shape should be the number of fish receptors by number of time points.'
        elif len(E_field.shape) == 3:
            transdermal_signal  = (self.receptors_normals[:,:,np.newaxis] * E_field).sum(1) * self.skin_rho * water_conductivity
            assert (transdermal_signal.shape[0] == self.receptors_locations.shape[0]) and (transdermal_signal.shape[1] == E_field.shape[2]), 'Electric image perturbation shape should be the number of fish receptors by number of time points.'
        else:
            raise ValueError('Electric field input should be shape (N x 3) or (N x 3 x T).')
        return transdermal_signal
    

    def compute_receptors_responses(self, transdermal_signal):
        assert self.eod_length == self.receptor_filters.shape[1], "The receptor length must match the eod length for proper computation."
        responses = np.inner(transdermal_signal[:,-self.eod_length:], self.receptor_filters)
        return responses


    def update_receptors(self, receptors_locations=None, receptors_normals=None):
        ''' Update the receptors of the fish, i.e. move the fish in space. '''
        self.receptors_locations        = self.receptors_locations       if receptors_locations is None else receptors_locations
        self.receptors_normals          = self.receptors_normals         if receptors_normals   is None else receptors_normals
        assert self.receptors_locations.shape[0]       == self.receptors_normals.shape[0],        'Number of receptors locations and normals must match.'
        assert (self.receptors_locations.shape[1] == 3) and (self.receptors_normals.shape[1] == 3), 'All locations and normals must be 3-dimensional.'
        if (self.receptors_locations.shape[0] == 0) or (self.receptors_normals.shape[0] == 0):
            print('Fish receptors were not properly innitialized - some have 0 length.')
    

    def update_point_currents(self, point_currents_mag=None, point_currents_loc=None):
        ''' Update the point currents of the fish, i.e. move the fish in space. '''
        self.point_currents_magnitudes  = self.point_currents_magnitudes if point_currents_mag  is None else point_currents_mag
        self.point_currents_locations   = self.point_currents_locations  if point_currents_loc  is None else point_currents_loc
        assert self.point_currents_magnitudes.shape[0] == self.point_currents_locations.shape[0], 'Number of point currents locations and magnitudes must match.'
        assert self.point_currents_locations.shape[1] == 3, 'All locations must be 3-dimensional.'
        if (self.point_currents_magnitudes.shape[0] == 0) or (self.point_currents_locations.shape[0] == 0):
            print('Fish point currents were not properly innitialized - some have 0 length.')

    
    def update_eod_waveform_and_delay(self, eod_waveform=None, sampling_rate=None, eod_delay=None):
        self.sampling_rate = self.sampling_rate                   if sampling_rate is None else convert2mainSI(sampling_rate)
        self.eod_delay     = 0                                    if eod_delay     is None else convert2mainSI(eod_delay)
        self.eod_waveform  = self.eod_waveform[-self.eod_length:] if eod_waveform  is None else eod_waveform
        self.eod_length    = self.eod_length                      if eod_waveform  is None else eod_waveform.shape[0]
        self.eod_wave_max  = np.array(self.eod_waveform).max()
        # include the eod delay at the begining of the wave-form
        self.eod_wave_form = np.hstack([np.zeros(np.int64(self.sampling_rate * self.eod_delay)), np.array(self.eod_waveform) / (self.eod_wave_max if self.eod_wave_max > 0 else 1)])
        self.time_stamps   = np.arange(self.eod_wave_form.shape[0]) / self.sampling_rate
    

    def update_receptors_filters(self, new_receptor_filters):
        self.receptor_filters = new_receptor_filters


    def is_equal(self, other, simple_return=True):
        ''' Compare current worm to another worm for equality. '''
        _, thruth_values, comparison  = super().is_equal(other, simple_return=False)
        if isinstance(other, self.__class__):
            thruth_values.append(np.abs(self.ID                - other.get_ID()) < self.assert_err)
            thruth_values.append(np.abs(self.sampling_rate     - other.get_sampling_rate()) < self.assert_err)
            thruth_values.append(np.abs(self.skin_rho          - other.get_skin_resistivity()) < self.assert_err)

            comparison.append('ID')
            comparison.append('sampling_rate')
            comparison.append('skin_resistivity')
        thruth_value = False if len(thruth_values) == 0 else np.array(thruth_values).all()
        if simple_return:
            return thruth_value  
        else:   
            return thruth_value, thruth_values, comparison


    def visualize_scatter(self, fig=None, intensity=None, cbar_N_ticks=10,
                          marker=None, marker_size=10, marker_alpha=0.8,
                          color_map='Viridis', fig_width=None, fig_height=None, units_prefix='m',
                          xaxis_title='X AXIS TITLE', yaxis_title='Y AXIS TITLE', zaxis_title='z AXIS TITLE',
                          show_normals=5, show_point_currents=30, update_layout=True):
        ''' 
        Visualize the fish as scatter points. If provided, the intensity of the 
        perturbation is reflected in the color of each scatter point.
        '''
        if fig is None:
            fig = go.Figure()

        if show_point_currents:
            fig.add_trace(self.create_point_currents_graph_obj(marker_size=show_point_currents, units_prefix=units_prefix))

        if show_normals:
            fig.add_trace(self.create_normals_graph_obj(size_scale=show_normals, units_prefix=units_prefix))

        intensity_range = None if intensity is None else [intensity.min(), intensity.max()]
        fig.add_trace(self.create_scatter_graph_obj(intensity=intensity, intensity_range=intensity_range, units_prefix=units_prefix, marker_size=marker_size, 
                                                    color_map=color_map, marker_alpha=marker_alpha, cbar_N_ticks=cbar_N_ticks))
        
        if update_layout:
            fig.update_layout(scene = dict(xaxis = dict(backgroundcolor="rgb(220, 220, 240)",
                                                        gridcolor="white",
                                                        showbackground=True,
                                                        zerolinecolor="white",
                                                       ),
                                           yaxis = dict(backgroundcolor="rgb(240, 220, 240)",
                                                        gridcolor="white",
                                                        showbackground=True,
                                                        zerolinecolor="white",
                                                       ),
                                           zaxis = dict(backgroundcolor="rgb(240, 240, 220)",
                                                        gridcolor="white",
                                                        showbackground=True,
                                                        zerolinecolor="white",
                                                       ),
                                           xaxis_title=xaxis_title + ' (%sm)'%units_prefix,
                                           yaxis_title=yaxis_title + ' (%sm)'%units_prefix,
                                           zaxis_title=zaxis_title + ' (%sm)'%units_prefix,
                                           aspectmode='data', # aspectratio=dict(x=1, y=1, z=2),
                                          ),
                               width=fig_width,
                               height=fig_height,
                               margin=dict( r=0, l=0, b=0, t=0),
                              )
        fig.show()
        return fig


    def create_scatter_graph_obj(self, intensity=None, intensity_range=None, units_prefix='', marker_size=10, color_map='Viridis', marker_alpha=0.7, cbar_N_ticks=10):
        ''' Create a plotly graphical object to insert into a plot at a later time. This GO shows the fish as scattered points. '''
        if intensity is None:
            colors = self.receptors_locations[:,0]  # np.ones(self.receptors_locations.shape[0])
        else:
            colors = intensity
        if intensity_range is None:
            intensity_range = [colors.min(), colors.max()]
            if colors.shape[0] > 1:
                colors = (colors - colors.min()) / colors.ptp() 
        else:
            colors = (colors - intensity_range[0]) / (intensity_range[1]-intensity_range[0])

        graph_obj = go.Scatter3d(x=add_prefix(self.receptors_locations[:,0], units_prefix), 
                                 y=add_prefix(self.receptors_locations[:,1], units_prefix), 
                                 z=add_prefix(self.receptors_locations[:,2], units_prefix), 
                                 mode='markers', showlegend=False,
                                 marker=dict(size=marker_size,
                                             color=colors,
                                             colorscale=color_map,
                                             opacity=marker_alpha,
                                             colorbar=dict(len=0.8, tickmode='array', tickvals=np.linspace(0,1,cbar_N_ticks),
                                                           ticktext=np.linspace(0,1,cbar_N_ticks) if intensity_range is None else np.linspace(intensity_range[0], intensity_range[1], cbar_N_ticks),
                                                           ),
                                             cmin=0, cmax=1,
                                              )
                                  )
        return graph_obj


    def create_normals_graph_obj(self, size_scale=5, units_prefix=''):
        ''' Create a plotly graphical object to insert into a plot at a later time. This GO shows normals on the fish. '''
        graph_obj = go.Cone(x=add_prefix(self.receptors_locations[:,0], units_prefix),
                            y=add_prefix(self.receptors_locations[:,1], units_prefix),
                            z=add_prefix(self.receptors_locations[:,2], units_prefix),
                            u=self.receptors_normals[:,0],
                            v=self.receptors_normals[:,1],
                            w=self.receptors_normals[:,2],
                            colorscale='Blues',
                            sizemode="absolute",
                            sizeref=size_scale,
                            showscale=False, showlegend=False,)
        return graph_obj


    def create_point_currents_graph_obj(self, marker_size=15, units_prefix=''):
        ''' Create a plotly graphical object to insert into a plot at a later time. This GO shows point currents within the fish. '''
        graph_obj = go.Scatter3d(x=add_prefix(self.point_currents_locations[:,0], units_prefix), 
                                 y=add_prefix(self.point_currents_locations[:,1], units_prefix), 
                                 z=add_prefix(self.point_currents_locations[:,2], units_prefix), 
                                 mode='markers', showlegend=False,
                                 marker=dict(size=marker_size,
                                             color=self.point_currents_magnitudes,
                                             colorscale='Bluered',
                                             opacity=1,
                                            )
                                )
        return graph_obj


    def visualize_triangulation(self):
        raise NotImplementedError('Triangulation visualization of fish not yet implemented - easier to export to Matlab...')


    def details(self):
        ''' Provides the details of the object. Can be adjusted in child classes to include more information. '''
        details_string  =  super().details()
        details_string +=(f'This fish has:\n'\
                          f'skin resistivity {self.skin_rho} (unit: Ohm * meter^2)\n'\
                          f'conductivity {self.sig} (unit: Siemes / meter) / relative permitivity {self.eps_r}\n'\
                          f'EOD wave form sampling rate {self.sampling_rate} (unit: Hertz)\n'\
                          f'number of point currents {self.point_currents_magnitudes.shape[0]} / number of receptors: {self.receptors_locations.shape[0]}\n')
        return details_string


    def get_ID(self):
        return self.ID


    def get_receptors_locations(self):
        return self.receptors_locations


    def get_receptors_normals(self):
        return self.receptors_normals


    def get_sampling_rate(self):
        return self.sampling_rate


    def get_eod_wave_form(self):
        return self.eod_wave_form


    def get_time_stamps(self):
        return self.time_stamps


    def get_skin_resistivity(self):
        return self.skin_rho


    def get_point_currents_magnitude(self):
        return self.point_currents_magnitudes


    def get_point_currents_location(self):
        return self.point_currents_locations


    def get_N_receptors(self):
        return self.receptors_locations.shape[0]
    

    def get_receptors_filters(self):
        return self.receptor_filters
    

    def get_N_filters(self):
        return self.receptor_filters.shape[0]


    @classmethod
    def _initialize_input_argument_names(cls):
        inp_args  = super()._initialize_input_argument_names()
        inp_args += ['receptors_locations=np.array([]).reshape(0,3)',
                       'receptors_normals=np.array([]).reshape(0,3)',
                       'point_currents_magnitudes=np.array([]).reshape(0)',
                       'point_currents_locations=np.array([]).reshape(0,3)',
                       'receptor_filters=np.array([]).reshape(0,1000)',
                       'eod_waveform=np.arange(1000)',
                       'skin_resistivity=1',
                       'sampling_rate=(2.5, "M"),  # MHz',
                       'eod_delay=(0,''), # s',]
        return inp_args


    def run_tests(self):
        ''' Sanity assertion checks to ensure code robustness. '''
        super().run_tests()
        # assert self.skin_sig        == self.get_skin_conductivity(), 'Fish skin conductivity does not match.'
        assert self.skin_rho        == self.get_skin_resistivity(), 'Fish skin resistivity does not match.'
        assert self.sampling_rate   == self.get_sampling_rate(), 'Fish sampling rate does not match.'

        assert (self.receptors_locations         == self.get_receptors_locations()).all(), 'Fish receptors locations do not match.'
        assert (self.receptors_normals           == self.get_receptors_normals()).all(), 'Fish receptors normals do not match.'

        assert (self.time_stamps    == self.get_time_stamps()).all(), 'Fish time stamps does not match.'
        assert (self.eod_wave_form  == self.get_eod_wave_form()).all(), 'Fish eod wave form does not match.'
        assert (self.point_currents_magnitudes   == self.get_point_currents_magnitude()).all(), 'Fish point currents magnitude does not match.'
        assert (self.point_currents_locations    == self.get_point_currents_location()).all(), 'Fish point currents location does not match.'

        return 'Success!'


