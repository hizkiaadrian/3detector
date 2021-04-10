from cv2 import erode, dilate
from numpy import ones

def eroded_mask(depth_map, kernel=ones((6,12))):
  return (erode(depth_map, kernel) != -1)

def dilated_mask(depth_map, kernel=ones((6,12))):
  return (dilate(depth_map, kernel) != -1)

def loss1(reference_dm, rendered_dm):
  mask = eroded_mask(reference_dm)

  return sum_of_squares(reference_dm[mask], rendered_dm[mask])

def loss2(reference_dm, rendered_dm):
  inner_mask = eroded_mask(reference_dm)
  outer_mask = ~dilated_mask(reference_dm)

  mask = inner_mask | outer_mask

  return sum_of_squares((reference_dm != -1)[mask].astype(int), (rendered_dm != -1)[mask].astype(int))

def sum_of_squares(arr1, arr2):
  return ((arr1 - arr2) ** 2).sum()