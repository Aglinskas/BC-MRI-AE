cd '/Users/aidasaglinskas/Desktop/BC-MRI-AE/matlab_code'
addpath('/Users/aidasaglinskas/Desktop/BC-MRI-AE/matlab_code')
%%

group = 'TB'
table = readtable(sprintf('../CSVs/corner_%s.csv',group));
subs_high = table.high;
subs_low = table.low;

jacobian_fn_temp = '../Assets/jacobians/normed_Js_masked/%s_normed_Js_masked.nii';

scans1 = cellfun(@(x) sprintf(jacobian_fn_temp,x),subs_high,'UniformOutput',false);
scans2 = cellfun(@(x) sprintf(jacobian_fn_temp,x),subs_low,'UniformOutput',false);
analysis_dir = sprintf('../SPMs/analysis_%s',group);

spm('defaults','fMRI');
spm_jobman('initcfg')
matlabbatch = make_batch(analysis_dir,scans1,scans2);
save(sprintf('matlabbatch_%s.mat',group),'matlabbatch')
spm_jobman('run',matlabbatch)

%%