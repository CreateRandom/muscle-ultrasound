clear; clc
cd('\\umcczzoknf01\Data_knf\research_data_echo\auto_lines')
out_path = '\\umcczzoknf01\Data_knf\research_data_echo\auto_lines\initial_patients';
out_path_archive = '\\umcczzoknf01\Data_knf\research_data_echo\auto_lines\archive_patients';
rootdir = '\\umcczzoknf01\Data_knf\Qumia_data';
rootdir_archive = '\\umcczzoknf01\Data_knf\Workroom\Archive';
lp_path = 'labeled patients_deduped.mat';
lp_path_missing = '\\umcczzoknf01\Data_knf\research_data_echo\auto_lines\initial_patients\missing_patients.mat';

% patient folder regex: 7-digit ID, year, date, month, hour, minute
reg = '(\d{7})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})';
reg_archive = '(\d{7})\.(\d{4})(\d{2})(\d{2})\.(\d{2})(\d{2})';

use_archive = false;
copy_images = false;
if use_archive
    rootdir = rootdir_archive;
    lp_path = lp_path_missing;
    reg = reg_archive;
    out_path = out_path_archive;
end

% labeledpatients = readtable(lp_path);
load(lp_path,'labeledpatients');

disp('Found a total of ' + string(size(labeledpatients,1)) + ' labeled patients.');

patient_list = dir(rootdir);
T = struct2table(patient_list);
sortedT = sortrows(T, 'datenum');
patient_list = table2struct(sortedT);

available_pids = {};
match_count = 1;
for patient = 1:size(patient_list,1)
    % try to parse the name using the regex
    folder_name = patient_list(patient).name;
    match = regexp(folder_name,reg,'tokens');
    if ~isempty(match)
        match = match{1};
        available_pids{match_count, 1} = match{1};
        available_pids{match_count, 2} = folder_name;
        timestamp = datetime(str2double(match{2}),str2double(match{3}),str2double(match{4}), str2double(match{5}), str2double(match{6}),0);
        available_pids{match_count, 3} = timestamp;
        match_count = match_count + 1;
    end
    
end
disp('Found a total of ' + string(size(available_pids,1)) + ' patient ids.');
% convert to string array to be able to more easily access
labeledpatients.pid = string(labeledpatients.pid);

missing_patients = setdiff(labeledpatients.pid,available_pids(:,1));
disp('Found a total of ' + string(size(missing_patients,1)) + ' labelled patients that are missing.');

% find duplicate patients
pids = available_pids(:,1);
[v, w] = unique( pids, 'stable' );
duplicate_indices = setdiff( 1:numel(pids), w);
duplicates = unique(pids(duplicate_indices));
disp('Found a total of ' + string(size(duplicates,1)) + ' patients with more than one result.');
labeled_dups = intersect(duplicates,labeledpatients.pid);
disp(string(size(labeled_dups,1)) + ' of those patients are labeled currently.');
a=1;

% find the ids we need
available_pids = cell2table(available_pids,'VariableNames',{'pid', 'folder_name', 'date'});
available_pids.pid = string(available_pids.pid);

% convert to string array to be able to more easily access
labeledpatients.pid = string(labeledpatients.pid);

unique_patients_to_copy = intersect(available_pids.pid,labeledpatients.pid);

counter = 1;
missing_counter = 1;
missing_xml = {};
dup_counter = 1;
dup_results = {};
% suppress warning about existing directories
warning('off','MATLAB:MKDIR:DirectoryExists');
% fields that can be found in the results xml
xml_fields3 = {'muscle_name', 'side', 'EI', 'EI_normal', 'EI_zscore', 'thickness', 'thickness_normal','thickness_zscore'};
xml_fields2_1 = {'EI', 'validnormal', 'EIzscore','normal', 'name', 'side','musclekey'};

patient_fields3 = {'pid', 'Name', 'Birthdate', 'Height', 'Weight', 'Sex', 'Age','RecordingDate', 'Side', 'ReleaseDate'};
patient_fields2_1 = {'pid', 'Name', 'Birthdate', 'Height', 'Weight', 'Sex', 'Age','RecordingDate', 'Side'};
for patient = 1:size(unique_patients_to_copy,1)
    display([num2str(patient) '/' num2str(size(unique_patients_to_copy,1))]);
    
    id_to_copy = unique_patients_to_copy(patient);
    % get the records to copy
    records = available_pids(available_pids.pid == id_to_copy,:);
    % first check if each record has an associated xml file
    results_paths = fullfile(rootdir,records.folder_name,'results.xml');
    for x = 1:size(results_paths,1)
        % files that don't exist
        results_paths{x} = (exist(results_paths{x,:},'file') == 0);
    end
    to_remove = cell2mat(results_paths);
    records(to_remove,:) = [];
    
    if size(records,1) == 0
        disp('No result xml found for patient ' + id_to_copy + '. Skipping.');
        missing_xml{missing_counter} = id_to_copy;
        missing_counter = missing_counter + 1;
        continue;
    elseif size(records,1) == 1
        record = records(1,:); 
    else
        [a, i] = min(records.date);
        % picking the oldest one for now --> discuss
        record = records(i,:);
        disp('Picked older record for patient ' + id_to_copy);
        dup_counter = dup_counter + 1;
        dup_results{dup_counter, 1} = id_to_copy;
        dup_results{dup_counter, 2} = records;
    end
    folder_name = record.folder_name{1};
    target_path = fullfile(out_path, folder_name);
    
    if ~(exist(target_path,'dir') == 2)
        mkdir(target_path);
    end
    
    % patient path
    f = fullfile(rootdir,folder_name);
    % roi subdirectory
    f_roi = fullfile(rootdir,folder_name,'roi');
    % file with info
    % this contains muscles, sides and dcmfiles
    anal_file = load(fullfile(f_roi,'anal.mat'));

    image_table = [anal_file.muscles , anal_file.sides, anal_file.dcmfiles'];
    image_table = cell2table(image_table,  'VariableNames',{'Muscle', 'Side', 'DCM'});
    % remove DCMs with no muscle name (e.g. fasciculation vids)
    loc=cellfun(@isempty, image_table{:,'Muscle'} );
    image_table(loc,:)=[];

    writetable(image_table,fullfile(target_path,'images.xlsx'))
    patient_char = anal_file.patient;
    % info from the xml on muscle, not image level
    s = xml2struct(fullfile(f,'results.xml'));

    if isfield(s,'qumia_xml')
        muscle_struct = s.qumia_xml.muscle;
        patient_struct = s.qumia_xml.patient;
        ref_fields = xml_fields3;
        legacy=false;
        patient_fields = patient_fields3;
    else
        muscle_struct = s.qumiaroot.metingen.muscle;
        patient_struct = s.qumiaroot.patient;
        ref_fields = xml_fields2_1;
        legacy=true;
        patient_fields = patient_fields2_1;
    end

    %muscle_cells = [];
    muscles = [];
    for muscle_id = 1:size(muscle_struct,2)
        muscle = muscle_struct{muscle_id};
        % standardize to the same fields
        fields = fieldnames(muscle);
        % drop muscles without a score
        if ~ismember('EI',fields)
            continue
        end
        to_remove = ~ismember(fields, ref_fields);
        muscle = rmfield(muscle,fields(to_remove));
        muscles{muscle_id} = FullyFlattenStruct(muscle);
        %muscle_cell = flattenStruct2Cell(muscle);
        %muscle_cells{muscle_id} = muscle_cell;
    end
    % one entry for each recorded muscle (and side) with scores and
    % raw values
    merged_muscles = mergeStructs(muscles{:});
    muscle_table = struct2table(merged_muscles);

    %muscle_cells = vertcat(muscle_cells{:});
    %muscle_table = cell2table(muscle_cells, 'VariableNames', ref_fields);

    writetable(muscle_table,fullfile(target_path,'muscles.xlsx'))
    % some rare images don't have a side info, delete those
    no_side_label=all(cellfun(@isempty,image_table{:,2}),2);
    if sum(no_side_label) > 0
        image_table(no_side_label,:)=[];
        disp('Deleted image for which no side information was available');
    end
    % rename some variable for consistency
    if legacy
        muscle_table.Properties.VariableNames{'name_Text'} = 'Muscle';
    else
        muscle_table.Properties.VariableNames{'muscle_name_Text'} = 'Muscle';
    end
    muscle_table.Properties.VariableNames{'side_Text'} = 'Side';
    % join the tables to get the muscle info for each image
    % image_table = innerjoin(image_table, muscle_table);
    image_table = outerjoin(image_table, muscle_table, 'Keys',{'Muscle','Side'}, 'Type', 'Left', 'MergeKeys', true);
    % get the patient info into a table as well
    patient_info = flattenStruct2Cell(patient_struct);
    patient_info = cell2table(patient_info, 'VariableNames', patient_fields);
    % overwrite the ID because sometimes it just says FLIK in the
    % file
    patient_info.pid = {id_to_copy};
    % remove personal information
    patient_info.Name = [];
    patient_info.Birthdate = [];

    class_label = labeledpatients(labeledpatients.pid == patient_info.pid{1},:).class;
    patient_info.Class = class_label;
    % fill in the patient level info
    image_table.pid = repelem(string(id_to_copy),size(image_table,1))';
    image_table.folder_name = repelem(string(record.folder_name{1}),size(image_table,1))';
    image_table.Age = repelem(string(patient_info.Age{1}),size(image_table,1))';
    image_table.Height = repelem(string(patient_info.Height{1}),size(image_table,1))';
    image_table.Weight = repelem(string(patient_info.Weight{1}),size(image_table,1))';
    image_table.Sex = repelem(string(patient_info.Sex{1}),size(image_table,1))';
    image_table.Class = repelem(string(patient_info.Class{1}),size(image_table,1))';
    image_tables{counter} = image_table;
    patient_infos{counter} = patient_info;
    
    counter = counter +1;

    image_list = image_table.DCM;

    image_list = dir(fullfile(f_roi,'*dcm.mat'));

    if copy_images
        for k=1:size(image_list,2)
                % read in the original DICOM (this throws out all
                % tabular information, including the patient name)
                I = dicomread(fullfile(f,image_list{k}));
                % new path
                new_path = fullfile(target_path, image_list{k});
                dicomwrite(I,new_path);
                % copy over the roi annotation
                new_roi_path = fullfile(target_path,[image_list{k} '.mat']);
                copyfile(fullfile(f,'roi',[image_list{k} '.mat']),new_roi_path)
        end
    end
end
patient_infos = vertcat(patient_infos{:});
writetable(patient_infos,fullfile(out_path,'patients.xlsx'))
total_table = vertcat(image_tables{:});
writetable(total_table,fullfile(out_path,'full_format_image_info.xlsx'))

all_missing = [string(missing_patients); missing_xml'];
inds = ismember(labeledpatients.pid, all_missing);
missing_patients = labeledpatients(inds, :);
% this messes the table up, leading 0s get deleted
writetable(missing_patients,fullfile(out_path,'missing_patients.xlsx'))
% instead, save to a matfile, rename to allow reading in again directly
labeledpatients = missing_patients;
save(fullfile(out_path,'missing_patients.mat'), 'labeledpatients');

dup_results = vertcat(dup_results{:,2});
writetable(dup_results,fullfile(out_path,'multi_record_patients.xlsx'))