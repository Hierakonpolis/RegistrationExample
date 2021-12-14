#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import ants, time, copy
import numpy as np

# def normalize(x):
#     return (x - x.mean()) / x.std()
def slice_img(img, idx, direction):
    return ants.from_numpy(img[:,:,:,idx], origin = img.origin[:3], spacing = img.spacing[:3], direction=direction)

def mode(array, axis=None):
    vals, counts = np.unique(array, return_counts=True, axis = axis)
    mode_value = np.argwhere(counts == np.max(counts))
    return vals[mode_value]

class dataset:
    def __init__(self,root_folder,
                 scan_name='raw_phMRIscan.nii.gz',
                 default_motion_correction=None,
                 mot_corr_name='motioncorr.nii.gz',
                 default_registration=None,
                 reg_name='registered_to_template.nii.gz'):
        
        # crawl the dataset root folder and collect the paths of the files
        # by setting here standard names for all files, we can
        # check if they have already been generated and index them,
        # or simply store those names for later use
        self.mot_corr_name = mot_corr_name
        self.root = root_folder
        self.default_motion_correction = default_motion_correction
        self.regname = reg_name
        self.default_registration = default_registration
        motcorr_scans = []
        regscans = []
        scans = []
        # index each folder to find the paths we are interested in
        for dirpath, dirnames, filenames in os.walk(root_folder):
            if scan_name in filenames:
                scans.append(os.path.join(dirpath,scan_name))
                motcorfile = os.path.join(dirpath,mot_corr_name)
                regfile = os.path.join(dirpath,reg_name)
                if os.path.isfile(regfile):
                    regscans.append(regfile)
                if os.path.isfile(motcorfile):
                    motcorr_scans.append(motcorfile)
        self.regscans = regscans
        self.motcorr_scans = motcorr_scans
        self.scans = scans
        self.template = os.path.join(root_folder, 'template.nii.gz')
        self.template_mask = os.path.join(root_folder, 'template_mask.nii.gz')
        self.read = ants.image_read
        
        # If we only found some registered scans, and not others, warn the user
        if len(regscans) > 0 and len(regscans) != len(scans):
            print('Registered scans are missing for some dataset items')
        
        # same for motion correction
        if len(motcorr_scans) > 0 and len(motcorr_scans) != len(scans):
            print('Motion-corrected scans are missing for some dataset items')    
        
        
    # methods so object can be indexed
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self,index):
        return self.scans[index]
    
    # motion correct all samples
    def motioncorrect_all(self):
        residuals = []
        # have we defined standard parameters?
        if self.default_motion_correction is None:
            raise NotImplementedError('Motion correction parameters not defined')
        
        
        motcorr_scans = []
        for x in self:
            # get the image
            img = ants.image_read(x)
            # get the parameters
            pars = copy.deepcopy(self.default_motion_correction)
            pardir = os.path.dirname(x)
            # set volume series
            pars['image'] = img
            # do motion correction
            mc = ants.motion_correction(**pars)
            # remember the error
            residuals.append(mc['FD'])
            
            # save output
            resultimage = mc['motion_corrected']
            motcorr_file = os.path.join(pardir, self.mot_corr_name)
            ants.image_write(resultimage, motcorr_file)
            motcorr_scans.append(motcorr_file)
        # save list of volumes
        self.motcorr_scans = motcorr_scans
        # return the errors
        return residuals
            
    
    def prel_outliers_report(self, array, name):
        # a function to signal outliers for a spacing or shape dimension
        # it will also print some general stats dataset_wide
        outliers_index = array - array.mean(0) > array.std(0)*2
        sample, item = np.where(outliers_index)
        print('Average',name, np.mean(array,0))
        print(name,'standard deviation', np.std(array,0))
        print(name, 'median:', np.median(array,0))
        print(name, 'mode:', mode(array,0))
        print('Outliers:')
        for s, i in zip(sample,item):
            path = self[s]
            print('Outlier in sample', path)
            print(name,'on axis', i, '=', array[s,i])
        
        # this tells us whether we found anything at all
        found = np.sum(outliers_index)
        if found == 0: print('No', name, 'outliers found')
            
        
    def check_scan_params(self):
        
        # checks for outlier on basic formatting anomalies
        shapes = []
        spacings = []
        
        for impath in self:
            image = ants.image_read(impath)
            shapes.append(image.shape)
            spacings.append(image.spacing)
        
        # the following lines will implicitly return an error if 
        # different volumes have an inconsistent number of dimensions
        spacings = np.array(spacings)
        shapes = np.array(shapes)
        
        # let's look for outliers
        self.prel_outliers_report(spacings, 'spacing')
        self.prel_outliers_report(shapes, 'shape')
        
        self.shapes = shapes
        self.spacings = spacings
        
        return spacings, shapes
    
    def motion_correction_error(self,
                                function_parameters,
                                subjects=None
                                ):
        # refer to the comments for motion correction for all scans
        # this applies it to a subset measuring runtime, returning no image
        
        start = time.time()
        if subjects is None:
            subjects = range(len(self))
        FDs = []
        for index in subjects:
            x = ants.image_read(self[index])
            fp = copy.deepcopy(function_parameters)
            fp['image'] = x
            FDs.append(ants.motion_correction(**fp)['FD'].mean())
        duration = (time.time() - start) / len(subjects)
        
        return np.mean(FDs), duration
    
    def calibrate_motion_correction(self,
                                    parameter_set,
                                    subjects=None):
        # check the motion correction error for different sets of parameters
        results = []
        for params in parameter_set:
            results.append(self.motion_correction_error(params, subjects))
        return results
    
    def registration_error(self,
                           function_parameters,
                           subjects=None
                           ):
        # registration error and time for a parameter configuration
        
        start = time.time()
        template = ants.image_read(self.template)
        template_mask = ants.image_read(self.template_mask)
        template_mask = template.new_image_like(template_mask[:])
        
        if subjects is None:
            subjects = range(len(self))
        errors = []
        for index in subjects:
            pardir = os.path.dirname(self[index])
            x = ants.image_read(os.path.join(pardir,self.mot_corr_name))
            fp = copy.deepcopy(function_parameters)
            mov = slice_img(x,0, template.direction)
            fp['fixed'] = template
            fp['moving'] = mov
            fp['mask'] = template_mask
            fp['imagetype'] = 3
            registration = ants.registration(**fp)
            
            transfrmd = ants.apply_transforms(template,
                                              x,
                                              transformlist = registration['fwdtransforms'],
                                              imagetype=3
                                              )
            err = - ants.image_mutual_information(slice_img(transfrmd,0,template.direction), template)
            errors.append(err)
            
            
        duration = (time.time() - start) / len(subjects)
        
        return np.mean(err), duration
    
    def calibrate_registration(self,
                               parameter_set,
                               subjects=None):
        # registration error and time for a set of parameter configurations
        results = []
        for params in parameter_set:
            results.append(self.registration_error(params, subjects))
        return results
    
    def register_all(self):
        # applies registration to the template space for all series
        self.regscans = []
        errs = []
        par = self.default_registration
        # parameters defined?
        if par is None:
            raise NotImplementedError('Registration parameters not defined')
        # Hey! The spacing metadata for the template mask was wrong! >://
        template = ants.image_read(self.template)
        template_mask = ants.image_read(self.template_mask)
        template_mask = template.new_image_like(template_mask[:])
        
        # for each series
        subjects = range(len(self))
        for index in subjects:
            pardir = os.path.dirname(self[index])
            # x: the series
            x = ants.image_read(os.path.join(pardir,self.mot_corr_name))
            fp = copy.deepcopy(par)
            # target image: the first volume in the series
            mov = slice_img(x,0, template.direction)
            # do the registration
            fp['fixed'] = template
            fp['moving'] = mov
            fp['mask'] = template_mask
            fp['imagetype'] = 3
            registration = ants.registration(**fp)
            # apply the transform
            transfrmd = ants.apply_transforms(template,
                                              x,
                                              transformlist = registration['fwdtransforms'],   #  fwdtransforms invtransforms
                                              imagetype=3
                                              )
            # save file path
            regfile = os.path.join(pardir,self.regname)
            # save file to memory
            ants.image_write(transfrmd, regfile)
            self.regscans.append(regfile)
            # errors to report
            err = - ants.image_mutual_information(slice_img(transfrmd,0,template.direction), template)
            errs.append(err)
            
        return errs




