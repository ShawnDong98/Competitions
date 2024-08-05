def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class RSNA24TestDataset(Dataset):
    def __init__(self, root_dir, df, study_ids, image_shape, phase='test', transform=None):
        self.root_dir = root_dir
        self.df = df
        self.study_ids = study_ids
        self.image_shape = image_shape
        self.transform = transform
        self.phase = phase


    def __len__(self):
        return len(self.study_ids)
    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob(f'{self.root_dir}/test_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (self.image_shape[0], self.image_shape[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(self.image_shape[0], self.image_shape[1])
        return img

    def __getitem__(self, idx):
        x = np.zeros(self.image_shape, dtype=np.uint8)
        st_id = self.study_ids[idx] 

        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass

        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')