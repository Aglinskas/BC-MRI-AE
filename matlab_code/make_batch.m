function matlabbatch = make_batch(analysis_dir,scans1,scans2 )
% analysis_dir = 'gi'
% scans1 = 
% scans2 = 

assert(all(cellfun(@exist,scans1)),'scans1 not found')
assert(all(cellfun(@exist,scans2)),'scans1 not found')

if exist(analysis_dir)==0
    disp('no dir - creating')
    mkdir(analysis_dir)
elseif exist(analysis_dir)==7
    if length(dir(analysis_dir))>2
        disp('is dir - emptying')
        pause(3)
        delete(fullfile(analysis_dir,'/*'))
    end
end

matlabbatch{1}.spm.stats.factorial_design.dir = {analysis_dir};
%%
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = scans1;
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = scans2;
%%
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1; %1
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;% 0
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;% 0
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = '1 -1';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = '-1 1';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 0;

matlabbatch{4}.spm.stats.results.spmmat(1) = cfg_dep('Contrast Manager: SPM.mat File', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{4}.spm.stats.results.conspec(1).titlestr = '';
matlabbatch{4}.spm.stats.results.conspec(1).contrasts = 1;
matlabbatch{4}.spm.stats.results.conspec(1).threshdesc = 'none';
matlabbatch{4}.spm.stats.results.conspec(1).thresh = 0.001;
matlabbatch{4}.spm.stats.results.conspec(1).extent = 0;
matlabbatch{4}.spm.stats.results.conspec(1).conjunction = 1;
matlabbatch{4}.spm.stats.results.conspec(1).mask.none = 1;
matlabbatch{4}.spm.stats.results.conspec(2).titlestr = '';
matlabbatch{4}.spm.stats.results.conspec(2).contrasts = 2;
matlabbatch{4}.spm.stats.results.conspec(2).threshdesc = 'none';
matlabbatch{4}.spm.stats.results.conspec(2).thresh = 0.001;
matlabbatch{4}.spm.stats.results.conspec(2).extent = 0;
matlabbatch{4}.spm.stats.results.conspec(2).conjunction = 1;
matlabbatch{4}.spm.stats.results.conspec(2).mask.none = 1;
matlabbatch{4}.spm.stats.results.units = 1;
matlabbatch{4}.spm.stats.results.export{1}.ps = true;
matlabbatch{4}.spm.stats.results.export{2}.jpg = true;
matlabbatch{4}.spm.stats.results.export{3}.pdf = true;
matlabbatch{4}.spm.stats.results.export{4}.csv = true;
matlabbatch{4}.spm.stats.results.export{5}.tspm.basename = datestr(datetime,'dd-mmm-yyyy-HH-MM-SS');
end



