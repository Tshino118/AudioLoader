

def run(self, file_info: PathInfo):
    self.__file_info = file_info
    data, sr = librosa.load(file_info.path)
    tr_data = torch.tensor(data)
    self.__source_info = {'rate': sr}

    for aug_list in AugType.AUG_LIST:
        if self.only_librosa(aug_list):
            self.run_aug_librosa(data, aug_list)
        else:
            self.run_aug(tr_data, aug_list)

def create_filename(self, aug_list):
    ret = []
    for aug_type in aug_list:
        aug_info = AugType.get_type(aug_type)
        ret.append(aug_info)
    dir_part = "_".join(ret)
    if ret == []:
            return self.__file_info.get_increment_path()
    self.__file_info.add_dir_part(dir_part)
    return self.__file_info.get_increment_path()

def run_aug(self, data, aug_list):
    for augment_type in aug_list:
        data = self.execute_aug(data, augment_type)
    file_name = self.create_filename(aug_list)
    torchaudio.save(file_name, data, self.__source_info["rate"])

def run_aug_librosa(self, data, aug_list):
    for augment_type in aug_list:
        data = self.execute_aug_librosa(data, augment_type)
    file_name = self.create_filename(aug_list)
    write(file_name, self.__source_info["rate"], data)

def execute_aug(self, data, aug_type):
    if "noise" == AugType.get_type(aug_type):
        return self.apply_noise(data)
    elif "time" == AugType.get_type(aug_type):
        return self.apply_time(data)
    elif "pitch" == AugType.get_type(aug_type):
        return self.apply_pitch(data)
    elif "band" == AugType.get_type(aug_type):
        return self.apply_band(data)
    elif "clip" == AugType.get_type(aug_type):
        return self.apply_clip(data)
    elif "reverb" == AugType.get_type(aug_type):
        return self.apply_reverb(data)
    else:
        raise ValueError(f"Wrong aug_type value:{aug_type}")

def execute_aug_librosa(self, data, aug_type):
    if "noise" == AugType.get_type(aug_type):
        return self.noise_injection(data)
    elif "time" == AugType.get_type(aug_type):
        return self.shifting_time(data)
    elif "pitch" == AugType.get_type(aug_type):
        return self.changing_pitch(data)
    elif "speed" == AugType.get_type(aug_type):
        return self.changing_speed(data)
    else:
        raise ValueError(f"Wrong aug_type value:{aug_type}")

def only_librosa(self, aug_type_list):
    key_value = [k for k, v in AugType.AUG_TYPE.items() if v == 'speed'][0]
    if key_value in aug_type_list:
        return True
    return False
