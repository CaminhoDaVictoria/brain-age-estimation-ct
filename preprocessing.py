import os
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage as ndi
import SimpleITK as sitk
import cv2
from scipy.ndimage import zoom
import json
import matplotlib.pyplot as plt

# Paths
base_dir = "" #Path to the folder containing the DICOM images; enter here
label_csv = os.path.join(base_dir, ".csv") # Enter the name of the tagging CSV file here; ideally, it should be located in the same folder as the DICOM images

# Get labels 
labels_df = pd.read_csv(label_csv, sep=";") 

# Preprocessing: DICOM images
'''
Loads a DICOM series with full geometric awareness as a 3D volume.

For CT DICOM data, individual slices must not simply be stacked as
2D images using os.listdir. Correct 3D geometry depends on several
important DICOM fields, including:

- ImageOrientationPatient: Orientation of the image axes in patient space
- ImagePositionPatient: Position of each slice in patient space
- PixelSpacing: Pixel size in mm
- SpacingBetweenSlices / SliceThickness: Slice spacing and thickness
- GantryDetectorTilt: Indicates gantry tilt

SimpleITK/GDCM reconstructs a spatially correct 3D image from this
information. Afterwards, the image can optionally be resampled to a
uniform target spacing. This is the stage where differences in gantry
tilt and image orientation should be handled consistently.

Parameters
----------
dicom_dir : str
    Directory containing the DICOM files of a series.

target_spacing : tuple(float, float, float) or None
    Target voxel size in mm as (x, y, z). Example: (1.0, 1.0, 1.0).
    If None, the original DICOM spacing is preserved.

return_sitk : bool
    If True, the SimpleITK image is additionally returned.
    This is useful if further geometric operations will be
    performed in SimpleITK.

return_metadata : bool
    If True, a dictionary containing geometry-related metadata
    is additionally returned.

series_id : str or None
    Optional specific SeriesInstanceUID. If None, the largest
    detected series is used. This helps avoid selecting
    scout/localizer images.

verbose : bool
    If True, important geometry information is printed.

Returns
-------
volume : np.ndarray
    3D array in (z, y, x) order, dtype float32.

image : sitk.Image, optional
    Only returned when return_sitk=True.

metadata : dict, optional
    Only returned when return_metadata=True.
'''

def load_dicom_series(
    dicom_dir,
    target_spacing=(1.0, 1.0, 1.0),
    return_sitk=False,
    return_metadata=False,
    series_id=None,
    verbose=True
    ):


    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(dicom_dir) or [])

    if not series_ids:
        print(f"No DICOM series found in {dicom_dir}")
        return None

# If multiple series are found in the directory, the series containing
# the largest number of files is selected by default. This reduces the risk
# of accidentally loading scout/localizer images.
    if series_id is None:
        series_id = max(
            series_ids,
            key=lambda sid: len(reader.GetGDCMSeriesFileNames(dicom_dir, sid))
        )

    file_names = list(reader.GetGDCMSeriesFileNames(dicom_dir, series_id))
    if not file_names:
        print(f"No files found for series {series_id} in {dicom_dir}")
        return None

    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        image = reader.Execute()
        image = sitk.Cast(image, sitk.sitkFloat32)
    except Exception as e:
        print(f"Error loading DICOM series from {dicom_dir}: {e}")
        return None

    metadata = {
        "series_id": series_id,
        "num_files": len(file_names),
        "original_size_xyz": image.GetSize(),
        "original_spacing_xyz": image.GetSpacing(),
        "origin_xyz": image.GetOrigin(),
        "direction": image.GetDirection(),
        "target_spacing_xyz": target_spacing,
    }

# GantryDetectorTilt alone is not sufficient.
# The actual image geometry is defined by image orientation, image position, and voxel spacing information.
    try:
        ds0 = pydicom.dcmread(file_names[0], force=True, stop_before_pixels=True)
        metadata.update({
            "gantry_detector_tilt": float(getattr(ds0, "GantryDetectorTilt", 0.0)),
            "image_orientation_patient": list(getattr(ds0, "ImageOrientationPatient", [])),
            "image_position_patient_first": list(getattr(ds0, "ImagePositionPatient", [])),
            "pixel_spacing": list(getattr(ds0, "PixelSpacing", [])),
            "slice_thickness": getattr(ds0, "SliceThickness", None),
            "spacing_between_slices": getattr(ds0, "SpacingBetweenSlices", None),
        })
    except Exception as e:
        metadata["metadata_warning"] = str(e)

    if verbose:
        print("DICOM geometry:")
        print(f"  Serie: {metadata['series_id']}")
        print(f"  Dateien: {metadata['num_files']}")
        print(f"  Original size (x,y,z): {metadata['original_size_xyz']}")
        print(f"  Original spacing (x,y,z): {metadata['original_spacing_xyz']}")
        print(f"  GantryDetectorTilt: {metadata.get('gantry_detector_tilt')}")
        print(f"  ImageOrientationPatient: {metadata.get('image_orientation_patient')}")

    if target_spacing is not None:
        image = resample_sitk_image(
            image=image,
            target_spacing=target_spacing,
            interpolator=sitk.sitkLinear,
            default_value=-1024.0
        )
        metadata["resampled_size_xyz"] = image.GetSize()
        metadata["resampled_spacing_xyz"] = image.GetSpacing()

    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    outputs = [volume]
    if return_sitk:
        outputs.append(image)
    if return_metadata:
        outputs.append(metadata)

    return outputs[0] if len(outputs) == 1 else tuple(outputs)

'''
Resamples a SimpleITK 3D image to a specified voxel spacing.

This resampling is performed in physical space while preserving the
image origin and direction. As a result, it is significantly better
suited for DICOM geometry and gantry tilt handling than applying
2D cv2.resize independently to each slice.

For CT intensity data: use sitk.sitkLinear.
For segmentation masks or label images: use sitk.sitkNearestNeighbor.
'''
def resample_sitk_image(
    image,
    target_spacing=(1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
    default_value=-1024.0
    ):
        original_spacing = image.GetSpacing()  # (x, y, z) in mm
        original_size = image.GetSize()        # (x, y, z) in Voxeln

        target_size = [
            max(1, int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing(tuple(float(x) for x in target_spacing))
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(float(default_value))

        return resampler.Execute(image)

## Preprocessing Functions
'''
Returns only the largest connected 3D component of a binary mask.

This prevents small isolated regions caused by noise, residual bone
structures, or boundary artifacts from being treated as part of the
brain region.
'''
def _largest_connected_component(mask):
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return mask

    sizes = ndi.sum(mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes) + 1)
    return labeled == largest_label

'''
Removes non-brain tissue from a CT volume using a coarse segmentation mask.

Voxels outside the mask are not set to 0. In CT imaging, 0 HU corresponds
to water or soft tissue and therefore does not represent a true background
value after windowing. Instead, a value of -1024 HU is used, which
approximately corresponds to air. After windowing, the background is
therefore mapped cleanly to 0.

Parameters
----------
volume : np.ndarray
    CT volume in Hounsfield Units (HU), shape (z, y, x).

hu_threshold : tuple
    HU range used to generate the initial brain mask.

background_value : float
    Value assigned to voxels outside the brain mask.
    For CT data, -1024 HU is typically used.

closing_iterations : int
    Number of morphological closing iterations used to
    fill small holes and gaps in the mask.

return_mask : bool
    If True, the binary brain mask is additionally returned.
'''
def skull_stripping(
    volume,
    hu_threshold=(-100, 100),
    background_value=-1024.0,
    closing_iterations=1,
    return_mask=False
    ):

    vol = np.asarray(volume, dtype=np.float32)

    mask = (vol >= hu_threshold[0]) & (vol <= hu_threshold[1])
    if closing_iterations > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iterations)
        mask = ndi.binary_fill_holes(mask)

    brain_mask = _largest_connected_component(mask)

    stripped = vol.copy()
    stripped[~brain_mask] = background_value

    if return_mask:
        return stripped.astype(np.float32), brain_mask.astype(bool)
    return stripped.astype(np.float32)

'''
Performs rigid registration consisting of rotation and translation only,
without any deformable transformation.

This function operates on NumPy arrays. For workflows that require
strict preservation of patient geometry across multiple processing
steps, a fully SimpleITK-based pipeline using sitk.Image objects would
generally be preferable.

In this implementation, however, the most important DICOM geometric
information is already incorporated during image loading by resampling
the data onto a target grid. This should be mentioned in any publication
or technical documentation describing the processing pipeline.
'''

def rigid_registration(fixed_volume, moving_volume, default_value=-1024.0):
    fixed = sitk.GetImageFromArray(np.asarray(fixed_volume, dtype=np.float32))
    moving = sitk.GetImageFromArray(np.asarray(moving_volume, dtype=np.float32))

    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=100
    )
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    out_transform = registration_method.Execute(fixed, moving)

    resampled = sitk.Resample(
        moving,
        fixed,
        out_transform,
        sitk.sitkLinear,
        float(default_value),
        sitk.sitkFloat32
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)

'''
Performs deformable registration using a B-spline transformation.

For brain age prediction, deformable registration should be used with
caution because it can alter genuine age-related anatomical structures.
 Such transformations may inadvertently remove or distort biologically
 relevant information that contributes to the prediction task.

For this reason, deformable registration is included as a comparison
variant. In dataset versions that incorporate elastic registration,
the transformation is intentionally enabled and should be evaluated
separately. This provides an opportunity to assess whether anatomical
normalization improves performance or unintentionally removes
age-related features, an aspect that should be discussed in any
publication describing the methodology.
'''

def elastic_registration(
    fixed_volume,
    moving_volume,
    grid_size=(4, 4, 4),
    downsample_factor=0.25,
    max_iterations=50,
    verbose=True,
    enabled=False
    ):

    if not enabled:
        return np.asarray(moving_volume, dtype=np.float32)

    if downsample_factor < 1.0:
        fixed_low = zoom(fixed_volume, downsample_factor, order=1)
        moving_low = zoom(moving_volume, downsample_factor, order=1)
    else:
        fixed_low, moving_low = fixed_volume, moving_volume

    fixed = sitk.GetImageFromArray(np.asarray(fixed_low, dtype=np.float32))
    moving = sitk.GetImageFromArray(np.asarray(moving_low, dtype=np.float32))

    transform = sitk.BSplineTransformInitializer(fixed, grid_size)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=max_iterations
    )
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    out_transform = registration_method.Execute(fixed, moving)

    resampled = sitk.Resample(
        sitk.GetImageFromArray(np.asarray(moving_volume, dtype=np.float32)),
        sitk.GetImageFromArray(np.asarray(fixed_volume, dtype=np.float32)),
        out_transform,
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32
    )
    return sitk.GetArrayFromImage(resampled).astype(np.float32)

'''
Optional array-based resampling.

Under normal circumstances, this step is no longer required because
load_dicom_series already performs geometry-aware resampling to the
target spacing using SimpleITK/GDCM.

If new_spacing=None, the volume is returned unchanged.

This prevents a second blind resampling step that would not have access
to the original DICOM geometry information.
'''

def resample_volume(volume, new_spacing=None, current_spacing=None):
    if new_spacing is None:
        return np.asarray(volume, dtype=np.float32)

    if current_spacing is None:
        raise ValueError(
            "current_spacing must be specified when a NumPy array is resampled again."
        )

    img = sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32))
    img.SetSpacing(tuple(float(x) for x in current_spacing))

    return sitk.GetArrayFromImage(
        resample_sitk_image(img, target_spacing=new_spacing, default_value=-1024.0)
    ).astype(np.float32)

'''
Applies slice-wise Non-Local Means denoising without permanently altering
the Hounsfield Unit (HU) scale.

For compatibility with OpenCV, each slice is temporarily scaled to
uint8 for denoising. After processing, the data are transformed back
to the original HU range.

This approach preserves physically meaningful HU values while still
benefiting from the noise reduction capabilities of Non-Local Means
filtering.
'''

def denoising(volume, hu_range=(-1000, 1000), h=7, template_window_size=7, search_window_size=21):
    vol = np.asarray(volume, dtype=np.float32)
    hu_min, hu_max = hu_range
    out = np.empty_like(vol, dtype=np.float32)

    clipped = np.clip(vol, hu_min, hu_max)
    scaled = ((clipped - hu_min) / (hu_max - hu_min) * 255.0).astype(np.uint8)

    for i in range(vol.shape[0]):
        denoised_uint8 = cv2.fastNlMeansDenoising(
            scaled[i],
            None,
            h,
            template_window_size,
            search_window_size
        )
        out[i] = denoised_uint8.astype(np.float32) / 255.0 * (hu_max - hu_min) + hu_min

    return out.astype(np.float32)

'''
Applies CT windowing and normalizes the result to the range [0, 1].

For brain CT imaging, a narrower brain window is generally more
appropriate than a very wide bone window, as it provides better
contrast for intracranial soft tissue structures and age-related
anatomical features.
'''

def windowing(volume, hu_min=-100, hu_max=100):
    vol = np.clip(np.asarray(volume, dtype=np.float32), hu_min, hu_max)
    vol = (vol - hu_min) / float(hu_max - hu_min)
    return vol.astype(np.float32)

'''
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
slice-by-slice to data that have already been normalized to the range [0, 1].

CLAHE can improve local image contrast and enhance subtle anatomical
details. However, it also modifies the original intensity distribution
and therefore alters image intensity statistics.
'''
def clahe(volume, clip_limit=2.0, tile_grid_size=(8, 8)):
    vol = np.asarray(volume, dtype=np.float32)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    vol_out = np.empty_like(vol, dtype=np.float32)

    vol_255 = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
    for i in range(vol.shape[0]):
        vol_out[i] = clahe_obj.apply(vol_255[i]).astype(np.float32) / 255.0

    return vol_out.astype(np.float32)

'''
Crops, pads, and optionally resamples a 3D volume for CNN training.

This function is intended for normalized volumes after windowing and/or
CLAHE processing. Background voxels are expected to have a value of 0,
which is why the default threshold is set to 0.01.
'''
def preprocess_volume(
    volume,
    mask=None,
    target_shape=(128, 128, 128),
    intensity_threshold=0.01,
    margin=5,
    resample=True,
    background_value=0.0
    ):
    vol = np.asarray(volume, dtype=np.float32).copy()

    if mask is not None:
        region = mask > 0
    else:
        region = vol > intensity_threshold

    if not np.any(region):
        print("Warning: No non-empty voxels found. The volume will be center-adjusted instead.")
        return pad_or_crop_center(vol, target_shape, background_value=background_value)

    coords = np.argwhere(region)
    zmin, ymin, xmin = np.maximum(coords.min(axis=0) - margin, 0)
    zmax, ymax, xmax = np.minimum(coords.max(axis=0) + margin + 1, vol.shape)

    vol = vol[zmin:zmax, ymin:ymax, xmin:xmax]
    if mask is not None:
        mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    if resample:
        zoom_factors = np.array(target_shape, dtype=np.float32) / np.array(vol.shape, dtype=np.float32)
        vol = zoom(vol, zoom_factors, order=1)
        if mask is not None:
            mask = zoom(mask.astype(np.uint8), zoom_factors, order=0).astype(bool)
    else:
        vol = pad_or_crop_center(vol, target_shape, background_value=background_value)
        if mask is not None:
            mask = pad_or_crop_center(mask.astype(np.uint8), target_shape, background_value=0).astype(bool)

    vol = np.clip(vol, 0.0, 1.0).astype(np.float32)
    return (vol, mask) if mask is not None else vol


'''
Centrally crops and/or pads a 3D volume to match the specified
target shape.
'''
def pad_or_crop_center(volume, target_shape, background_value=0.0):
    output = np.full(target_shape, background_value, dtype=volume.dtype)
    input_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)

    start_in = np.maximum(0, (input_shape - target_shape) // 2)
    end_in = np.minimum(input_shape, start_in + target_shape)

    start_out = np.maximum(0, (target_shape - input_shape) // 2)
    end_out = start_out + (end_in - start_in)

    output[start_out[0]:end_out[0],
           start_out[1]:end_out[1],
           start_out[2]:end_out[2]] =         volume[start_in[0]:end_in[0],
               start_in[1]:end_in[1],
               start_in[2]:end_in[2]]
    return output

'''
Removes small artifacts after normalization.

The input is expected to be a normalized volume with values in the
range [0, 1], where background voxels are close to 0.

Intensity values within the mask are preserved. Only voxels outside
the cleaned mask are set to 0.
'''
def regularisierung(volume, threshold=0.01, median_size=3, keep_largest_component=True):
    vol = np.asarray(volume, dtype=np.float32)
    filtered = ndi.median_filter(vol, size=median_size)

    mask = filtered > threshold
    mask = ndi.binary_fill_holes(mask)

    if keep_largest_component and np.any(mask):
        mask = _largest_connected_component(mask)

    cleaned = filtered.copy()
    cleaned[~mask] = 0.0
    return np.clip(cleaned, 0.0, 1.0).astype(np.float32)

'''
Central, reproducible preprocessing pipeline for brain age prediction
volumes.

Processing Order
----------------
1. Optional skull stripping in HU space, with background set to -1024 HU
2. Optional rigid registration
3. Optional elastic registration for comparison datasets
4. Optional HU-preserving denoising
5. Brain-window CT windowing and intensity normalization
6. Optional CLAHE
7. Crop and/or resample to the target CNN input shape
8. Mild regularization

This pipeline is designed to provide a standardized preprocessing
workflow while allowing controlled evaluation of the individual
processing steps and their impact on brain age prediction performance.
'''

def process_brain_age_volume(
    volume,
    apply_skull_stripping=True,
    apply_rigid_registration=False,
    fixed_reference_volume=None,
    apply_elastic_registration=False,
    apply_denoising=True,
    apply_clahe=True,
    target_shape=(128, 128, 128)
    ):

    vol = np.asarray(volume, dtype=np.float32)

    if apply_skull_stripping:
        vol = skull_stripping(vol, background_value=-1024.0)

    if apply_rigid_registration:
        if fixed_reference_volume is None:
            print("Rigid Registration skipped.")
        else:
            vol = rigid_registration(fixed_reference_volume, vol, default_value=-1024.0)

    if apply_elastic_registration:
        if fixed_reference_volume is None:
            print("Elastic Registration skipped.")
        else:
            vol = elastic_registration(fixed_reference_volume, vol, enabled=True)

    if apply_denoising:
        vol = denoising(vol)

    vol = windowing(vol, hu_min=-100, hu_max=100)

    if apply_clahe:
        vol = clahe(vol)

    vol = preprocess_volume(vol, target_shape=target_shape)
    vol = regularisierung(vol)

    return vol.astype(np.float32)



# Execution

'''
1. Skull Stripping
Removes non-brain tissue at the beginning of the pipeline so that
subsequent registration and resampling steps operate primarily on
brain structures.

2. Rigid Registration
Performs an initial coarse alignment. Applying this step after skull
stripping reduces potential influence from image boundaries and
extracranial tissues.

3. Elastic Registration
Useful when a finer anatomical alignment is required. For CT data,
deformable registration should be applied with caution, particularly
when no deformable ground-truth labels are available.

4. Volume Resampling
Interpolates the registered image data onto the target spacing.
Performing interpolation on the original intensity data generally
produces smoother and more consistent results.

5. Denoising
Applied after registration so that noise reduction operates on an
already aligned volume.

6. Windowing
Should be performed before CLAHE, as CLAHE is applied to normalized
intensity values.

7. CLAHE
Applied after windowing to enhance local contrast in a meaningful
intensity range.

8. preprocess_volume
Final cropping, padding, and optional resizing step that prepares the
volume for CNN input.

9. Regularization
Removes small artifacts and can fill minor cavities. When skull
stripping is applied, this step is often less critical but may still
help produce cleaner CNN inputs.

DS1: Full Preprocessing Pipeline
--------------------------------
The following preprocessing steps are applied:

* Skull stripping
* Rigid registration
* Elastic registration
* Resampling
* Denoising
* Windowing
* CLAHE
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing the processed dataset volumes
out_dir = os.path.join(base_folder, "DS1_Volumes") # Output location for the DS1 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM-folder not found: {dicom_dir}")
        continue

    print(f"\n---Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS1: Skull Stripping + Rigid Registration + Elastic Registration +
        # Denoising + Windowing + CLAHE + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=True,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=True,
            apply_denoising=True,
            apply_clahe=True,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds1.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main Meta file 
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds1.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Create a comprehensive metadata CSV for DS1 containing age, sex, slice count, and additional study information.
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir # Directory containing the numbered DICOM patient folders
output_base_folder = "ds_volumes" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS1_Volumes", "metadata_ds1.csv")
csv_out_path = os.path.join(output_base_folder, "DS1_Volumes", "patient_metadata_ds1.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert to a JSON- and CSV-compatible format---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


'''
Computes the slice normal vector from the ImageOrientationPatient
attribute.

For series with gantry tilt or non-perfectly axial orientations,
the z-coordinate alone is not sufficiently robust for determining
slice location.

The correct slice position is obtained by projecting
ImagePositionPatient onto the normal vector of the slice plane.
'''
def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


'''
Returns the slice position along the true slice-normal direction.

If ImageOrientationPatient is available, the slice location is computed
using the projection of ImagePositionPatient onto the slice normal vector.

As a fallback, ImagePositionPatient[2] is used when orientation
information is missing.
'''
def slice_position_along_normal(ds, normal):
    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
               # stop_before_pixels=True reads only the DICOM header information.
               # This is faster for metadata extraction and avoids unnecessary
               # decoding of large pixel arrays.
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
               # When stop_before_pixels=True is used, PixelData is intentionally not loaded.
               # Checking for Rows, Columns, and ImagePositionPatient provides a lightweight
               # header-based verification that the file represents an image slice.
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    # Sortierung entlang der echten Slice-Normalen statt nur Ã¼ber Z.
    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    # Slice-Abstand entlang der echten Normalen berechnen.
    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

'''
DS2: Preprocessing Pipeline Without Elastic Registration

The following preprocessing steps are applied:

* Skull stripping
* Rigid registration
* Resampling
* Denoising
* Windowing
* CLAHE
* preprocess_volume (cropping and padding)
* Regularization
''' 
base_folder = "" # Main directory containing the processed DS volume datasets
out_dir = os.path.join(base_folder, "DS2_Volumes") # Output location for the DS2 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Processed patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS2: Skull Stripping + Rigid Registration +
        # Denoising + Windowing + CLAHE + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=True,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=False,
            apply_denoising=True,
            apply_clahe=True,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds2.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds2.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS2 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS2_Volumes", "metadata_ds2.csv")
csv_out_path = os.path.join(output_base_folder, "DS2_Volumes", "patient_metadata_ds2.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert values to a JSON- and CSV-compatible format ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value

'''
Computes the slice normal vector from the ImageOrientationPatient
attribute.

For series with gantry tilt or non-perfectly axial orientations,
the z-coordinate alone is not sufficiently robust for determining
slice location.

The correct slice position is obtained by projecting
ImagePositionPatient onto the normal vector of the slice plane.
'''

def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm

'''
Returns the slice position along the true slice-normal direction.

If orientation information is available, the slice position is
computed along the actual slice normal vector.

Fallback: ImagePositionPatient[2] is used if orientation information
is missing.
'''
def slice_position_along_normal(ds, normal):
    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    # Sortierung entlang der echten Slice-Normalen statt nur Ã¼ber Z.
    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    # Compute slice spacing along the true slice-normal direction.
    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected patient metadata into a DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

'''
DS3: Preprocessing Pipeline Without Elastic Registration, CLAHE, and Skull Stripping

The following preprocessing steps are applied:

* Rigid registration
* Resampling
* Denoising
* Windowing
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS3_Volumes") # Output location for the DS3 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS3: Rigid Registration + Denoising + Windowing + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=False,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=False,
            apply_denoising=True,
            apply_clahe=False,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds3.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file 
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds3.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS3 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS3_Volumes", "metadata_ds3.csv")
csv_out_path = os.path.join(output_base_folder, "DS3_Volumes", "patient_metadata_ds3.csv")


# --- Helper-Funkcion: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert values to a JSON- and CSV-compatible format ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):
    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Alles in DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

''' 
DS4: Preprocessing Pipeline Without CLAHE

The following preprocessing steps are applied:

* Skull stripping
* Rigid registration
* Elastic registration
* Resampling
* Denoising
* Windowing
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS4_Volumes") # Output location for the DS4 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS4: Skull Stripping + Rigid Registration + Elastic Registration + Denoising + Windowing + CNN Shape Standardization

        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=True,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=True,
            apply_denoising=True,
            apply_clahe=False,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds4.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds4.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS4 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS4_Volumes", "metadata_ds4.csv")
csv_out_path = os.path.join(output_base_folder, "DS4_Volumes", "patient_metadata_ds4.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert values to a JSON- and CSV-compatible format ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):

    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):

    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

'''
DS5: Preprocessing Pipeline Without Elastic Registration and Skull Stripping

The following preprocessing steps are applied:

* Rigid registration
* Resampling
* Denoising
* Windowing
* CLAHE
* preprocess_volume (cropping and padding)
* Regularization
''' 
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS5_Volumes") # Output location for the DS5 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS5: Rigid Registration + Denoising + Windowing + CLAHE + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=False,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=False,
            apply_denoising=True,
            apply_clahe=True,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds5.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds5.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS5 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS5_Volumes", "metadata_ds5.csv")
csv_out_path = os.path.join(output_base_folder, "DS5_Volumes", "patient_metadata_ds5.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert metadata values into JSON- and CSV-compatible data types---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):

    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with calid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

''' 
DS6: Preprocessing Pipeline Without Skull Stripping

The following preprocessing steps are applied:

* Rigid registration
* Elastic registration
* Resampling
* Denoising
* Windowing
* CLAHE
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS6_Volumes") # Output location for the DS6 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS6: Rigid Registration + Elastic Registration + Denoising + Windowing + CLAHE + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=False,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=True,
            apply_denoising=True,
            apply_clahe=True,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds6.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds6.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS6 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Ordner mit den DS*_Volumes-Ausgabeordnern
volume_path = os.path.join(output_base_folder, "DS6_Volumes", "metadata_ds6.csv")
csv_out_path = os.path.join(output_base_folder, "DS6_Volumes", "patient_metadata_ds6.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert metadata values into JSON- and CSV-compatible data types ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):

    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

''' 
DS7: Preprocessing Pipeline Without Elastic Registration and CLAHE

The following preprocessing steps are applied:

* Skull stripping
* Rigid registration
* Resampling
* Denoising
* Windowing
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS7_Volumes") # Output location for the DS7 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS7: Skull Stripping + Rigid Registration + Denoising + Windowing + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.
        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=True,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=False,
            apply_denoising=True,
            apply_clahe=False,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds7.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds7.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Metad file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS7 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS7_Volumes", "metadata_ds7.csv")
csv_out_path = os.path.join(output_base_folder, "DS7_Volumes", "patient_metadata_ds7.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert metadata values into JSON- and CSV-compatible data types ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):
    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):
    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- CSVs mit Patienteninfos laden ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

''' 
DS8: Preprocessing Pipeline Without CLAHE and Skull Stripping

The following preprocessing steps are applied:

* Rigid registration
* Elastic registration
* Resampling
* Denoising
* Windowing
* preprocess_volume (cropping and padding)
* Regularization
'''
base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS8_Volumes") # Output location for the DS8 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS8: Rigid Registration + Elastic Registration + Denoising + Windowing + CNN Shape Standardization
        # fixed_volume represents the geometrically normalized reference volume
        # and serves as the baseline for all preprocessing comparison variants.

        vol = process_brain_age_volume(
            vol,
            apply_skull_stripping=False,
            apply_rigid_registration=True,
            fixed_reference_volume=fixed_volume,
            apply_elastic_registration=True,
            apply_denoising=True,
            apply_clahe=False,
            target_shape=(128, 128, 128)
        )
        out_path = os.path.join(out_dir, f"volume_{pid}_ds8.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds8.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS8 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS8_Volumes", "metadata_ds8.csv")
csv_out_path = os.path.join(output_base_folder, "DS8_Volumes", "patient_metadata_ds8.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert metadata values into JSON- and CSV-compatible data types ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):

    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):

    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- CSVs mit Patienteninfos laden ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

'''
DS9: No preprocessing
'''

base_folder = "" # Main directory containing all processed DS volume folders
out_dir = os.path.join(base_folder, "DS9_Volumes") # Output location for the DS9 volumes
os.makedirs(out_dir, exist_ok=True)
metadata = []

for idx, row in labels_df.iterrows():
    pid = row["Patient-Number"] # must align with labels
    age = row.get("Age", None) # must align with labels
    sex = row.get("Gender", None) # must align with labels
    dicom_dir = os.path.join(base_dir, str(pid))

    if not os.path.exists(dicom_dir):
        print(f"DICOM folder not found: {dicom_dir}")
        continue

    print(f"\n--- Process patient {pid} ({idx+1}/{len(labels_df)}) ---")
    vol = load_dicom_series(dicom_dir, target_spacing=(1.0, 1.0, 1.0))
    if vol is None:
        print(f"No valid volume for {pid}")
        continue

    orig_shape = vol.shape
    fixed_volume = vol.copy()

    try:
        # DS9 intentionally stores only the geometrically correct
        # DICOM-loaded and resampled volume in Hounsfield Units (HU).
        # This dataset serves as a control/baseline variant and does not
        # include any additional preprocessing steps.

        out_path = os.path.join(out_dir, f"volume_{pid}_ds9.npy")
        np.save(out_path, vol)

        metadata.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "input_shape": orig_shape,
            "processed_shape": vol.shape,
        })

        print(f"Finished: {pid} | saved volume: {vol.shape}")

    except Exception as e:
        print(f"Error for patient {pid}: {e}")


# ----------------------------
# Save main meta file
# ----------------------------
metadata_df = pd.DataFrame(metadata)
meta_path = os.path.join(out_dir, "metadata_ds9.csv")
metadata_df.to_csv(meta_path, index=False)
print(f"\n Meta file saved: {meta_path}")

# Generate a comprehensive metadata CSV for DS9 containing
# demographic and imaging information (e.g., age, sex, slice count).
# --- Root directory containing all patient data ---
dicom_base_dir = base_dir 
output_base_folder = "" # Directory containing the DS* volume output folders
volume_path = os.path.join(output_base_folder, "DS9_Volumes", "metadata_ds9.csv")
csv_out_path = os.path.join(output_base_folder, "DS9_Volumes", "patient_metadata_ds9.csv")


# --- Helper-Function: Safe access to DICOM attributes ---
def safe_get(ds, tag, default=None):
    return getattr(ds, tag, default)


# --- Helper-Function: Convert metadata values into JSON- and CSV-compatible data types ---
def make_json_compatible(value):
    if isinstance(value, (pydicom.multival.MultiValue, tuple, list, np.ndarray)):
        return list(value)
    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
        return float(value)
    elif value is None:
        return None
    else:
        return value


def slice_normal_from_orientation(ds):

    orientation = safe_get(ds, "ImageOrientationPatient", None)
    if orientation is None:
        return None

    orientation = np.asarray(orientation, dtype=np.float64)
    if orientation.size != 6:
        return None

    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None

    return normal / norm


def slice_position_along_normal(ds, normal):
    position = safe_get(ds, "ImagePositionPatient", None)
    if position is None:
        return None

    position = np.asarray(position, dtype=np.float64)
    if normal is None:
        return float(position[2])

    return float(np.dot(position, normal))


# --- Iterate through all patient folders ---
meta_list = []

for patient_name in sorted(os.listdir(dicom_base_dir)):
    patient_folder = os.path.join(dicom_base_dir, patient_name)
    if not os.path.isdir(patient_folder):
        continue

    print(f"Processing {patient_name}...")

    dicom_files = []
    for root, dirs, files in os.walk(patient_folder):
        for f in sorted(files):
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
                if hasattr(ds, "Rows") and hasattr(ds, "Columns") and hasattr(ds, "ImagePositionPatient"):
                    dicom_files.append(ds)
            except Exception as e:
                print(f"File {f} could not be read: {e}")

    if len(dicom_files) == 0:
        print(f"No valid DICOM files in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    dicom_files = [d for d in dicom_files if slice_position_along_normal(d, normal) is not None]
    dicom_files.sort(key=lambda d: slice_position_along_normal(d, normal))

    if len(dicom_files) == 0:
        print(f"No DICOM files with valid position in folder {patient_folder}")
        continue

    ds0 = dicom_files[0]
    normal = slice_normal_from_orientation(ds0)

    patient_info = {
        "PatientID": safe_get(ds0, "PatientID", "unknown"),
    }

    slice_thickness = [safe_get(ds, "SliceThickness") for ds in dicom_files]
    pixel_spacing = safe_get(ds0, "PixelSpacing", [None, None])
    orientation = safe_get(ds0, "ImageOrientationPatient", None)
    positions = [safe_get(ds, "ImagePositionPatient") for ds in dicom_files]

    tilts = [safe_get(ds, "GantryDetectorTilt") for ds in dicom_files if hasattr(ds, "GantryDetectorTilt")]
    mean_tilt = float(np.nanmean([float(t) for t in tilts if t is not None])) if len(tilts) > 0 else None

    projected_positions = [
        slice_position_along_normal(ds, normal)
        for ds in dicom_files
        if slice_position_along_normal(ds, normal) is not None
    ]
    mean_spacing_normal = (
        float(np.mean(np.abs(np.diff(sorted(projected_positions)))))
        if len(projected_positions) > 1 else None
    )

    meta_summary = {
        "Folder": patient_name,
        "PatientID": patient_info["PatientID"],
        "SliceCount": len(dicom_files),
        "SliceThickness_mean": float(np.nanmean([float(s) for s in slice_thickness if s is not None])) if any(s is not None for s in slice_thickness) else None,
        "SliceSpacing_Normal_Calculated": mean_spacing_normal,
        "PixelSpacing": pixel_spacing,
        "ImageOrientationPatient": orientation,
        "SliceNormal": normal.tolist() if normal is not None else None,
        "GantryDetectorTilt_mean": mean_tilt,
    }

    meta_summary_json = {k: make_json_compatible(v) for k, v in meta_summary.items()}
    meta_list.append(meta_summary_json)


# --- Convert all collected records into a pandas DataFrame ---
df = pd.DataFrame(meta_list)
print(f"{len(df)} patient folder successfully processed.")

# --- Load CSVs with patient information ---
volume_df = pd.read_csv(volume_path)

volume_df = volume_df.rename(columns={
    "patient_id": "Folder",
    "age": "PatientAge",
    "sex": "PatientSex"
})

# --- Merge ---
df["Folder"] = df["Folder"].astype(str)
volume_df["Folder"] = volume_df["Folder"].astype(str)
merged_df = df.merge(volume_df, on="Folder", how="right")

# --- Order of columns ---
cols = [
    "Folder", "PatientID", "PatientAge", "PatientSex",
    "GantryDetectorTilt_mean", "SliceCount", "SliceThickness_mean",
    "ImageOrientationPatient", "SliceNormal", "SliceSpacing_Normal_Calculated",
    "PixelSpacing", "input_shape", "processed_shape"
]
merged_df = merged_df[cols]

# --- Save ---
merged_df.to_csv(csv_out_path, index=False)
print(f"Meta files saved: {csv_out_path}")

