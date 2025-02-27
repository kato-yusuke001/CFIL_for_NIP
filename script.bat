#python learn_cfil.py --data_dir 20250127_almi --persam_f False
#python learn_cfil.py --data_dir 20250127_almi --persam_f True

#python learn_cfil.py --data_dir 20250127_blue --persam_f False
#python learn_cfil.py --data_dir 20250127_blue --persam_f True

#python learn_cfil.py --data_dir 20250129_black --persam_f False
#python learn_cfil.py --data_dir 20250129_black --persam_f True

#python learn_cfil.py --data_dir 20250129_mirror --persam_f False
#python learn_cfil.py --data_dir 20250129_mirror --persam_f True

#python learn_cfil.py --data_dir 20250129_white --persam_f False
#python learn_cfil.py --data_dir 20250129_white --persam_f True

python train_cfil.py --data_dir 20250127_almi --persam_f --mask_image_only --make_joblib
python train_cfil.py --data_dir 20250127_blue --persam_f --mask_image_only --make_joblib
python train_cfil.py --data_dir 20250129_black --persam_f --mask_image_only --make_joblib
python train_cfil.py --data_dir 20250129_mirror --persam_f --mask_image_only --make_joblib
python train_cfil.py --data_dir 20250129_white --persam_f --mask_image_only --make_joblib

python train_cfil.py --data_dir 20250127_almi --persam_f --mask_image_only --train
python train_cfil.py --data_dir 20250127_blue --persam_f --mask_image_only --train
python train_cfil.py --data_dir 20250129_black --persam_f --mask_image_only --train
python train_cfil.py --data_dir 20250129_mirror --persam_f --mask_image_only --train
python train_cfil.py --data_dir 20250129_white --persam_f --mask_image_only --train