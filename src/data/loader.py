import json;
import torch;
import scipy;
import os;
import numpy as np;
from tqdm import tqdm;
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import signal

from util_basis.Irreps import Irreps_build;
from util_basis.integral import integrate;

# Mapping from canonical delta-learning baseline name to the on-disk obs directory.
# Canonical names are lowercase; directory casing is preserved as stored.
# 'none' means "no baseline" (i.e., delta-learning from 0).
BASELINE_DIRS = {
    'b3lyp': 'obs_B3LYP',
    'bp86':  'obs_bp86',
}
VALID_BASELINES = set(BASELINE_DIRS) | {'none'}


def _normalize_baseline(name):
    if name is None:
        return 'none'
    key = str(name).strip().lower()
    if key not in VALID_BASELINES:
        raise ValueError(f"Unknown delta_baseline={name!r}. Expected one of {sorted(VALID_BASELINES)}")
    return key


def check_pt_file(path, device="cpu"):
    if not os.path.exists(path):
        return 'NotExist'   # file doesn't exist
    try:
        _ = torch.load(path, map_location=device)
        return 'Success'   # success
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return 'Fail'   # load failed


class dataloader():

    def __init__(self, device, element_list, 
                 path,  
                 starting_basis = 'def2-SVP'):
        
        self.preprocess_device = 'cpu';
        self.device = device
        self.path = path +'/';
        #self.batch_size = batch_size;
        irreps = Irreps_build(element_list);
        #self.elements = element_list;
        self.nbasis = irreps.get_nbasis();
        self.ne_dic = irreps.get_ne();
        self.op_names = ['x','y','z','xx','yy','zz','xy','xz','yz'];
        # Source subdir for the per-molecule "basic" feature files (Fock h, S, ...)
        # and the preprocessing cache. Default = BP86 ('basic'/'tmp'); set to
        # 'basic_B3LYP'/'tmp_basic_b3lyp' for the B3LYP-baseline experiment so the
        # caches never mix. (CCSD obs labels are functional-independent: still 'obs'.)
        self.basic_subdir = 'basic';
        self.tmp_subdir = 'tmp';

        self.integrator = integrate(device, starting_basis=starting_basis, path=path[:-7]);

    def _group_dir(self, file):
        """Group directory from a fl entry (file[0] ends in '/<basic_subdir>/')."""
        return file[0][:-(len(self.basic_subdir) + 2)]

    def read_basic(self, filepath):

        with open(filepath, 'r') as f:
            basic = json.load(f);
        pos = torch.tensor(basic['coordinates']).to(self.preprocess_device);
        pos = pos[:,[1,2,0]];
        elements = basic['elements'];

        #self.pos = pos;
        #self.elements = elements;
        nuclearCharge = [self.ne_dic[ele] for ele in elements];
        nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(self.preprocess_device);

        ne = int(round(sum([self.ne_dic[ele] for ele in elements])/2));
        norbs = int(round(sum([self.nbasis[ele] for ele in elements])));

        h = torch.tensor(basic['h']).to(self.preprocess_device);
        S_mhalf = scipy.linalg.fractional_matrix_power(basic['S'], (-1/2)).tolist();
        S_mhalf = torch.tensor(S_mhalf).to(self.preprocess_device);
        #self.S_mhalf = S_mhalf;
        h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);
        alpha_bp86 = torch.tensor(basic['alpha']).to(self.preprocess_device);

        return {'pos':pos,'elements':elements, 'S': torch.tensor(basic['S']).to(self.preprocess_device),
                'Smhalf': S_mhalf, 'h': h, 'E_nn':basic['Enn'],
                'ne':ne, 'norbs':norbs, 'nuclearCharge':nuclearCharge,'alpha_bp86':alpha_bp86};

    def read_baseline(self, filepath, nuclearCharge, elements):
        """Load a delta-learning baseline's observables (B3LYP, BP86, …).

        The on-disk schema is identical across supported baselines, so the
        parsing logic is baseline-agnostic — only the JSON file path differs.
        """
        with open(filepath, 'r') as f:
            obs = json.load(f)
        baseline = {}
        for op_name in ['x','y','z','xx','yy','zz','xy','xz','yz']:
            if op_name in obs:
                baseline[op_name] = obs[op_name]
        if 'alpha' in obs:
            baseline['alpha'] = torch.tensor(obs['alpha'], dtype=torch.float)
        if 'atomic_charge' in obs:
            baseline['atomic_charge'] = (-torch.tensor(obs['atomic_charge'], dtype=torch.float)
                                         + nuclearCharge)
        if 'bond_order' in obs:
            n = len(elements)
            B = torch.zeros(n, n)
            for dp in obs['bond_order']:
                B[dp[0], dp[1]] = dp[2]
                B[dp[1], dp[0]] = dp[2]
            baseline['B'] = B
        if 'Ee' in obs:
            baseline['E_gap'] = obs['Ee'] / 27.2114   # eV → Hartree
        if 'energy' in obs and 'E_nn' in obs:
            baseline['E'] = obs['energy'] - obs['E_nn']  # electronic energy for delta learning
        return baseline

    # Backward-compatible alias so any external callers keep working.
    def read_b3lyp(self, filepath, nuclearCharge, elements):
        return self.read_baseline(filepath, nuclearCharge, elements)

    def read_obs(self, filepath, pos,elements,S_mhalf,nuclearCharge):

        with open(filepath, 'r') as f:
            obs = json.load(f);

        label = {};

        if ('header' in obs):
            label['header'] = obs['header']
        else:
            label['header'] = None

        if('energy' in obs):
            label['E'] = obs['energy'];
            label['E_nn'] = obs['E_nn'];
        if('Ee' in obs):
            cm_inverse_to_hartree = 4.55633528*1E-6;
            try:
                label['E_gap'] = cm_inverse_to_hartree * obs['Ee'][0];
            except:
                raise ValueError(filepath)

        if('atomic_charge' in obs):
            label['atomic_charge'] = - torch.tensor(obs['atomic_charge']).to(self.preprocess_device) + nuclearCharge;

        if('bond_order' in obs):
            B_labels = torch.zeros([len(elements), 
                                    len(elements)]).to(self.preprocess_device);
            for dp in obs['bond_order']:
                B_labels[dp[0], dp[1]] = dp[2];
                B_labels[dp[1], dp[0]] = dp[2];
            label['B'] = B_labels;

        if('alpha' in obs):
            label['alpha'] = torch.tensor(obs['alpha']).to(self.preprocess_device);
        
        if('F' in obs):
            label['F'] = torch.tensor(obs['F']).to(self.preprocess_device);
        else:
            label['F'] = torch.zeros([len(obs['atomic_charge']),3]).to(self.preprocess_device);

        for op_name in self.op_names:
            if(op_name in obs):
                label[op_name] = obs[op_name]

        return label;
    
    def calc_charge_matrix(self, pos, elements, S_mhalf, S):
        irreps = Irreps_build(elements)
        nbasis = irreps.get_nbasis();
        orbitals_list = [nbasis[u] for u in elements];
        map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
        mati = [];
        for j in range(len(elements)):
            Sm = torch.zeros_like(S_mhalf);
            Sm[map1[j]:map1[j+1],:] = \
                S_mhalf[map1[j]:map1[j+1],:];
            S_res = torch.matmul(torch.matmul(S_mhalf, Sm),
                                S);
            mati.append(S_res);
        return torch.stack(mati)


    def read_obs_mat(self,pos,elements,S_mhalf,header):
        if header == None:
            center_type = "mass"
        else:
            #center_type = header["center"]
            center_type = "nuclei_charge"

        data_obs = {}
        for operator in self.op_names:
            res = self.integrator.calc_O(pos[None,:,:], 
                                         elements, operator,center = "nuclei_charge")[0];
            output = torch.matmul(torch.matmul(S_mhalf,
                                               torch.Tensor(res).to(self.preprocess_device)),
                                  S_mhalf)
            data_obs[operator] = output;

        Fmat = self.integrator.calc_F(pos, 
                                         elements);
        F_out = torch.matmul(torch.matmul(S_mhalf,
                                            torch.Tensor(Fmat).to(self.preprocess_device)),
                                S_mhalf)
        data_obs['F'] = F_out;

        return data_obs;

    
    def get_files(self, path_list, rank, world_size):

        fl = [];
        for path in path_list:
            fl += [(path, f) for f in os.listdir(path)];
        
        if(world_size == 1):

            return fl;
        
        else:

            files = fl[len(fl)*rank//world_size: len(fl)*(rank+1)//world_size]
            #partition = int(self.batch_size/world_size);

            return files;

    def load_data_serial(self, group, rank, world_size):
        
        path_list = [self.path + g + '/' + self.basic_subdir + '/' for g in group];
        fl = self.get_files(path_list, rank, world_size);
        
        data_in = [];
        labels = [];
        obs_mat = [];

        for file in tqdm(fl, desc="Loading data"):
        
            basic_path = file[0] + file[1];
            data_in.append(self.read_basic(basic_path));
            pos, elements, S_mhalf, nuclearCharge = data_in[-1]['pos'], data_in[-1]['elements'], data_in[-1]['Smhalf'], data_in[-1]['nuclearCharge'];
            obs_path = self._group_dir(file) + '/obs/' + file[1].split('_')[0]+file[1][-5:];
            labels.append(self.read_obs(obs_path,pos,elements,S_mhalf,nuclearCharge));
            obs_mat.append(self.read_obs_mat(pos,elements,S_mhalf,labels['header']));

        return data_in, labels, obs_mat,  fl;
  
    def pre_processing(self, group, rank, world_size, fl_override=None):
        path_list = [self.path + g + '/' + self.basic_subdir + '/' for g in group]
        fl = fl_override if fl_override is not None else self.get_files(path_list, rank, world_size)
        for g in group:
            os.makedirs(self.path + g + '/' + self.tmp_subdir + '/', exist_ok=True)

        def process_file(file):
            tmp_path = self._group_dir(file) + '/' + self.tmp_subdir + '/' + file[1].split('_')[0] + '.pt'
            if check_pt_file(tmp_path) == 'Success':
                return 0
            basic_path = file[0] + file[1]
            try:
                basic_data = self.read_basic(basic_path)
            except RuntimeError as e:
                import sys
                print(f"\n\n{'='*80}\n[ERROR] Failed processing file: {basic_path}\n  Error: {e}\n{'='*80}\n", file=sys.stderr, flush=True)
                raise

            pos, elements = basic_data['pos'], basic_data['elements']
            S_mhalf, nuclearCharge = basic_data['Smhalf'], basic_data['nuclearCharge']
            S = basic_data['S']

            obs_path = self._group_dir(file) + '/obs/' + file[1].split('_')[0] + file[1][-5:]
            label = self.read_obs(obs_path, pos, elements, S_mhalf, nuclearCharge)

            obs_data = self.read_obs_mat(pos, elements, S_mhalf,label['header'])
            charge_mat = self.calc_charge_matrix(pos, elements, S_mhalf, S)
            torch.save((basic_data, label, obs_data, charge_mat), tmp_path)
            #return basic_data, label, obs_data

        data_in, labels, obs_mat = [], [], []
        
        max_workers = 1 #max(1, os.cpu_count() - 2)  # Leave one CPU free
        print('start pre-processing with', max_workers, 'threads')
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, file) for file in fl]
    
                for f in tqdm(as_completed(futures), total=len(futures), desc="pre-processing data"):
                    f.result()
                    #basic_data, label, obs_data = f.result()  # exceptions show up here
                    #data_in.append(basic_data)
                    #labels.append(label)
                    #obs_mat.append(obs_data)
        except KeyboardInterrupt:
            print("\n❌ Interrupted by user")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

        #return data_in, labels, obs_mat, partition, fl
    
    def _baseline_cache_path(self, file, baseline):
        """Return (dir, pt_path) for the per-baseline observable cache."""
        group_dir = self._group_dir(file)   # strip trailing '/basic/'
        cache_dir = os.path.join(group_dir, f'tmp_baseline_{baseline}')
        pt_path = os.path.join(cache_dir, file[1].split('_')[0] + '.pt')
        return cache_dir, pt_path

    def _load_or_build_baseline(self, file, baseline, data_in_i, map_loc):
        """Load the per-baseline observables for one molecule.

        Uses a baseline-specific `tmp_baseline_<name>/` cache so we only parse
        the JSON once per baseline. Returns a dict of baseline observables on
        ``map_loc``, or None if the source JSON is missing.
        """
        obs_dir = BASELINE_DIRS[baseline]
        json_path = os.path.join(self._group_dir(file), obs_dir, file[1].split('_')[0] + '.json')
        cache_dir, pt_path = self._baseline_cache_path(file, baseline)

        if os.path.exists(pt_path):
            try:
                cached = torch.load(pt_path, map_location=map_loc)
                return cached
            except Exception as e:
                print(f"Failed to load baseline cache {pt_path}: {e} — rebuilding")

        if not os.path.exists(json_path):
            return None

        baseline_obs = self.read_baseline(json_path,
                                          data_in_i['nuclearCharge'].cpu(),
                                          data_in_i['elements'])
        os.makedirs(cache_dir, exist_ok=True)
        try:
            torch.save(baseline_obs, pt_path)
        except Exception as e:
            print(f"Warning: could not persist baseline cache {pt_path}: {e}")
        if map_loc != 'cpu':
            baseline_obs = {k: (v.to(map_loc) if isinstance(v, torch.Tensor) else v)
                            for k, v in baseline_obs.items()}
        return baseline_obs

    def load_data(self, group, rank, world_size, delta_learning=True,
                  delta_baseline='b3lyp', max_items=None, load_obs=True):
        """Load preprocessed molecule records.

        ``delta_baseline`` selects which delta-learning starting point to
        attach to each sample: ``'b3lyp'`` (default), ``'bp86'``, or ``'none'``
        (no baseline — learn absolute labels from 0). When ``delta_learning``
        is False, the baseline is forced to ``'none'``.
        """
        baseline = 'none' if not delta_learning else _normalize_baseline(delta_baseline)
        #data_in, labels, obs_mat, partition, fl = self.pre_processing(group, rank, world_size)
        path_list = [self.path + g + '/' + self.basic_subdir + '/' for g in group]
        fl = self.get_files(path_list, rank, world_size)
        if max_items is not None:
            fl = fl[:max_items]

        # Only skip preprocessing if the specific tmp files we are going to load exist.
        # (Checking only for tmp/ directory existence is insufficient and can lead
        # to FileNotFoundError when some .pt files are missing.)
        need_preprocess = False
        for file in fl:
            tmp_path = self._group_dir(file) + '/' + self.tmp_subdir + '/' + file[1].split('_')[0] + '.pt'
            if not os.path.exists(tmp_path):
                need_preprocess = True
                break

        if need_preprocess:
            self.pre_processing(group, rank, world_size, fl_override=fl)

        data_in, labels, obs_mat, charge_mat, kept_fl = [], [], [], [], []
        skipped_by_group = {}
        for file in tqdm(fl, desc="Loading data"):
            tmp_path = self._group_dir(file) + '/' + self.tmp_subdir + '/' + file[1].split('_')[0] + '.pt'
            #print(tmp_path)
            map_loc = self.device
            if type(map_loc) == int:
                map_loc = f"cuda:{self.device}"
            loaded = torch.load(tmp_path, map_location=map_loc if load_obs else 'cpu')
            data_in_i, labels_i, obs_mat_i, charge_mat_i = loaded
            if not load_obs:
                # Re-load only data_in and labels to GPU, discard obs/charge to save memory
                if map_loc != 'cpu':
                    for k, v in data_in_i.items():
                        if isinstance(v, torch.Tensor):
                            data_in_i[k] = v.to(map_loc)
                    for k, v in labels_i.items():
                        if isinstance(v, torch.Tensor):
                            labels_i[k] = v.to(map_loc)
                del obs_mat_i, charge_mat_i
                obs_mat_i, charge_mat_i = {}, {}
            # Attach the chosen delta-learning baseline observables, cached per-baseline.
            # If the per-sample baseline JSON is absent (e.g., the upstream parser
            # skipped the molecule for a non-physical BP86 gap), drop the sample
            # entirely so the sampler never sees a mixed batch.
            if baseline != 'none':
                baseline_obs = self._load_or_build_baseline(
                    file, baseline, data_in_i,
                    map_loc if load_obs else 'cpu')
                if baseline_obs is None:
                    mol_id = file[1].split('_')[0]
                    group_name = os.path.basename(self._group_dir(file))
                    skipped_by_group.setdefault(group_name, []).append(mol_id)
                    continue
                labels_i['baseline'] = baseline_obs
            data_in.append(data_in_i)
            labels.append(labels_i)
            obs_mat.append(obs_mat_i)
            charge_mat.append(charge_mat_i)
            kept_fl.append(file)

        if skipped_by_group:
            total_skipped = sum(len(v) for v in skipped_by_group.values())
            print(f"[loader] Dropped {total_skipped} molecule(s) lacking '{baseline}' baseline obs:")
            for g in sorted(skipped_by_group):
                ids = skipped_by_group[g]
                preview = ", ".join(ids[:10]) + (", ..." if len(ids) > 10 else "")
                print(f"  {g}: {len(ids)} dropped [{preview}]")
        fl = kept_fl
        '''
        # convert all torch tensors to the self.device device
        for d in data_in:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.to(self.device)
        for l in labels:
            for k, v in l.items():
                if isinstance(v, torch.Tensor):
                    l[k] = v.to(self.device)
        for o in obs_mat:
            for k, v in o.items():
                if isinstance(v, torch.Tensor):
                    o[k] = v.to(self.device)
        '''
        return data_in, labels, obs_mat, charge_mat, fl