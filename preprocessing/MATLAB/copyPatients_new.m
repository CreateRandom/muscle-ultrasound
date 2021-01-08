clear; clc
cd('\\umcczzoknf01\Data_knf\research_data_echo\auto_lines')
% the folder that will house the converted output
out_path = '\\umcczzoknf01\Data_knf\research_data_echo\auto_lines\total_patients';
% the currently used directory for saving patient records
current.path = '\\umcczzoknf01\Data_knf\Qumia_data';
% an archive with older patient records mostly from the Philips device
archive.path = '\\umcczzoknf01\Data_knf\Workroom\Archive';
% a folder with controls used for calibration
controls.path = 'H:\KNF\Research\Studies\Proefpersonen spierecho 2013-2014\Data metingen spierecho Esaote';

%lp_path = 'labeled patients_deduped.mat';
% a table with patient ids and diagnosis
% in the current setup, this just contains patients
% from the original label study (500), cramp and controls
lp_path = 'labeledPatients_cramp_controls.xlsx';

% also include patients that do not have a label yet
% as additional labels will be added later
% in the current setup, the labels provided by the medical student are added later
include_unlabeled = true;
% regular expressions for file name parsing differ by source

% patient folder regex: 7-digit ID, year, date, month, hour, minute
current.reg = '(\d{7})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})';
% archive regex 7-digit ID, year, date, month, hour, minute
archive.reg = '(\d{7})\.(\d{4})(\d{2})(\d{2})\.(\d{2})(\d{2})';
% the control patients don't conform to any scheme
controls.reg = '(.*)';

% policy for enforcing the presence of an xml file differs by source as well
% they often don't have an xml file, so we don't enforce it,
% fall back to the analysis file
controls.no_xml_enforced = true;
current.no_xml_enforced = false;
archive.no_xml_enforced = false;

locations = [current; archive; controls];
%locations = [controls];

% flags for omitting steps to speed up the process as needed
analyze_images = true;
copy_images = true;
% what format to export to
export_format = ".png";

% optionally, select only the record for each patient that has the highest
% number of core muscles, if false, use all records
pick_one_record_per_patient = false;

labeledpatients = readtable(lp_path);
%load(lp_path,'labeledpatients');
disp('Found a total of ' + string(size(labeledpatients,1)) + ' labeled patients.');
% now, go over all available records, unless available records have been cached
% make sure to delete this file to avoid use of cached records
% e.g. if new patient folders have become available
has_cache = exist('available_records.mat','file');
if ~has_cache
    disp('Parsing available records from disk.');
    available_recs = {};
    record_count = 1;

    for loc_ind = 1:size(locations,1)
        n_records_before = size(available_recs,1);
        % load in all the folder names from this location
        rootdir = locations(loc_ind).path;
        disp('Loading folder ' + string(rootdir));
        patient_list = dir(rootdir);
        T = struct2table(patient_list);
        sortedT = sortrows(T, 'datenum');
        patient_list = table2struct(sortedT);
        disp('Found a total of ' + string(size(patient_list,1)) + ' patient records.');
        % parse the folder names, exclude folders that do not match conventions
        reg = locations(loc_ind).reg;
        initial_record_count = size(available_recs,1);
        for patient = 1:size(patient_list,1)
            if mod(patient,1000) == 0
                disp('Processed record ' + string(patient));
            end
            full_path = fullfile(patient_list(patient).folder, patient_list(patient).name);
            % only proceed with folders
            if ~(exist(full_path,'dir') == 7)
               continue; 
            end
            folder_name = patient_list(patient).name;

            % try to parse the name using the regex
            match = regexp(folder_name,reg,'tokens');
            % see whether there is an associated results file
            results_paths = fullfile(rootdir,folder_name,'results.xml');
            results_exist = exist(results_paths,'file') == 2;
            analysis_path = fullfile(rootdir,folder_name,'roi', 'anal.mat');
            analysis_exists = exist(analysis_path,'file') == 2;
            % see what we need to enforce on this level.
            if locations(loc_ind).no_xml_enforced
               to_enforce = analysis_exists;
            else
               to_enforce = results_exist;
            end
            % only include if proper file name and analysis file exists
            if ~isempty(match) && to_enforce
                match = match{1};
                available_recs{record_count, 1} = match{1};
                available_recs{record_count, 2} = fullfile(rootdir,folder_name);
                if length(match) == 6
                    timestamp = datetime(str2double(match{2}),str2double(match{3}),str2double(match{4}), str2double(match{5}), str2double(match{6}),0);
                else
                    % fallback to folder metainfo
                    timestamp = datetime(patient_list(patient).date,'Locale','nl-nl');
                end
                available_recs{record_count, 3} = timestamp;
                record_count = record_count + 1;
            end
        end
    records_added = size(available_recs,1) - n_records_before;
    disp('Added ' + string(records_added) + ' valid patient records.'); 
    end
    save('available_records.mat', 'available_recs');
else
    disp('Loading records from cache.');
    available_records = load('available_records.mat');
    available_recs = available_records.available_recs;
end


disp('Found a total of ' + string(size(available_recs,1)) + ' valid patient records.');
% convert to string array to be able to more easily access
labeledpatients.pid = string(labeledpatients.pid);

missing_patients = setdiff(labeledpatients.pid,available_recs(:,1));
disp('Found a total of ' + string(size(missing_patients,1)) + ' labelled patients that are missing.');

% find duplicate patients
pids = available_recs(:,1);
[v, w] = unique( pids, 'stable' );
duplicate_indices = setdiff( 1:numel(pids), w);
duplicates = unique(pids(duplicate_indices));
disp('Found a total of ' + string(size(duplicates,1)) + ' patients with more than one result.');
labeled_dups = intersect(duplicates,labeledpatients.pid);
disp(string(size(labeled_dups,1)) + ' of those patients are labeled currently.');
a=1;

% find the ids we have available
available_recs = cell2table(available_recs,'VariableNames',{'pid', 'folder_name', 'date'});
available_recs.pid = string(available_recs.pid);

% convert to string array to be able to more easily access
labeledpatients.pid = string(labeledpatients.pid);

if include_unlabeled
    not_yet_labeled = setdiff(available_recs.pid,labeledpatients.pid);
    disp('Including ' + string(size(not_yet_labeled,1)) + ' patients not yet labeled.');
    nyl_table = table(not_yet_labeled,'VariableNames',{'pid'});
    nyl_table.class = repelem({NaN},size(nyl_table,1))';
    % nyl_table.class_number = repelem(NaN,size(nyl_table,1))';
    labeledpatients = vertcat(labeledpatients, nyl_table);
end

unique_patients_to_copy = intersect(available_recs.pid,labeledpatients.pid);

counter = 1;
missing_counter = 1;
missing_xml = {};
dup_counter = 1;
dup_results = {};
% suppress warning about existing directories
warning('off','MATLAB:MKDIR:DirectoryExists');
% fields that can be found in the results xml

% muscle fields
% xml_fields3 = {'muscle_name', 'side', 'EI', 'EI_normal', 'EI_zscore', 'thickness', 'thickness_normal','thickness_zscore'};
% xml_fields2_1 = {'EI', 'validnormal', 'EIzscore','normal', 'name', 'side','musclekey'};
% use a reduced set for consistency
xml_fields3 = {'muscle_name', 'side', 'EI', 'EI_zscore',};
xml_fields2_1 = {'EI', 'EIzscore', 'name', 'side'};

% patient: full fields one finds in the xml
patient_fields3 = {'pid', 'Name', 'Birthdate', 'Height', 'Weight', 'Sex', 'Age','RecordingDate', 'Side', 'ReleaseDate'};
patient_fields2_1 = {'pid', 'Name', 'Birthdate', 'Height', 'Weight', 'Sex', 'Age','RecordingDate', 'Side'};
patient_fields1_66 = {'pid', 'Birthdate', 'Height', 'Weight', 'Sex', 'Age','RecordingDate', 'Side'};

rec_counter = 1;
rec_versions = [];
% use this to be able to perform the process in smaller batches
% to avoid data loss when it's interrupted
% e.g. by setting start to 1 and end to 200 and so forth in batches
start_patient = 1;
end_patient = size(unique_patients_to_copy,1);
for patient = start_patient:end_patient
    display([num2str(patient) '/' num2str(size(unique_patients_to_copy,1))]);
    
    id_to_copy = unique_patients_to_copy(patient);
    % get the records to copy
    records = available_recs(available_recs.pid == id_to_copy,:);
    
    if size(records,1) > 1 && pick_one_record_per_patient
        n_muscles_max = 0;
        best_ind = 1;
        for rec_ind = 1:size(records,1)
            rec_path = records.folder_name{rec_ind};
            s = xml2struct(fullfile(rec_path,'results.xml'));
            if isfield(s,'qumia_xml')
                muscle_struct = s.qumia_xml.muscle; 
            else
                muscle_struct = s.qumiaroot.metingen.muscle;
            end
            % check the muscles contained in the structure
            [muscle_count,names, core_present] = checkMuscleStruct(muscle_struct);
            
            if core_present > n_muscles_max
                n_muscles_max = core_present;
                best_ind = rec_ind;
            end
        end
        % only retain this record
        records = records(best_ind,:);
        disp('Picked record with ' + string(n_muscles_max) + ' core muscles for patient ' + id_to_copy);
        dup_counter = dup_counter + 1;
        dup_results{dup_counter, 1} = id_to_copy;
        dup_results{dup_counter, 2} = records;
    end
    
%     for rec_ind = 1:size(records,1)
%         folder_name = records.folder_name{rec_ind};
%         s = xml2struct(fullfile(folder_name,'results.xml'));
%         q_version = classifyQumia(s);
%         if ~ismember(q_version, unique(rec_versions))
%            disp('Found new version')
%         end
%         
%         rec_versions(rec_counter) = q_version;
%         rec_counter = rec_counter + 1;
%     end

    % iterate over all the records here
    for rec_ind = 1:size(records,1)
    
        folder_name = records.folder_name{rec_ind}; %record.folder_name{1};
        % this chops of the end of the folder_name (after the dot) :/
        %[~,inner_folder,~] = fileparts(folder_name);
        C = strsplit(folder_name,'\');
        inner_folder = C{end};
        target_path = fullfile(out_path, inner_folder);
        % make the folder to copy the images to
        created_path = false;
        if ~(exist(target_path,'dir') == 2) && copy_images
            mkdir(target_path);
            created_path=true;
        end
        % copy only if we want to and we didn't have the path yet
        copy_images_cached = copy_images && created_path;
        % patient path
        f = folder_name;
        
        % Load the ROI mat file
        % roi subdirectory
        f_roi = fullfile(folder_name,'roi');
        % file with info
        % this contains muscles, sides and dcmfiles
        if ~exist(f_roi,'dir') || ~exist(fullfile(f_roi,'anal.mat'),'file')
            disp('Skipped ' + string(folder_name) + ', no ROI annotation.');
            continue;
        end
        
        
        % load the analysis file, this we really need
        anal_file = load(fullfile(f_roi,'anal.mat'));
        
        % load image information from the mat file in the ROI folder
        try
            % this works for newer files
            image_table = [anal_file.muscles , anal_file.sides, anal_file.dcmfiles'];
        catch
            try
                % sometimes, there's more dcm files than dcm files...
                if length(anal_file.dcmfiles) > length(anal_file.muscles)
                    disp('More DCM files than muscle names in mat file, cutting to shape');
                    % this should generally be safe
                    anal_file.dcmfiles = anal_file.dcmfiles(1:length(anal_file.muscles));
                end
                % necessary for some older files
                image_table = [anal_file.muscles' , anal_file.sides', anal_file.dcmfiles'];
            catch
                disp('Skipped ' + string(folder_name) + ', malformed muscle information, version ' + string(q_version));
                continue;
            end
        end
        
        image_table = cell2table(image_table, 'VariableNames',{'Muscle', 'Side', 'DCM'});
        % remove DCMs with no muscle name (e.g. fasciculation vids)
        loc=cellfun(@isempty, image_table{:,'Muscle'} );
        % remove DCMs starting with SC, these are not images, older patient
        % info slides (with names!)
        loc2 = cellfun(@(x) startsWith(x,'SC'),image_table{:,'DCM'});
        % same goes for DCMs ending in .0.dcm, newer patient info slides
        loc3 = cellfun(@(x) endsWith(x,'.0.dcm'),image_table{:,'DCM'});
        % and finally, there's also tif files in the field for some older
        % patients, so just toss out everything that's not dcm
        loc4 = cellfun(@(x) ~endsWith(x,'.dcm'),image_table{:,'DCM'});
        
        % remove either type
        image_table((loc | loc2 | loc3 | loc4),:)=[];
        
        % some rare images don't have a side info, delete those
        no_side_label=all(cellfun(@isempty,image_table{:,2}),2);
        if sum(no_side_label) > 0
            image_table(no_side_label,:)=[];
            disp('Deleted image for which no side information was available');
        end
        
        % check if the table is empty now, if so, skip over this patient
        % record
        if size(image_table,1) == 0
           disp('No dcm images found for folder ' + string(folder_name));
           continue; 
        end
        % writetable(image_table,fullfile(target_path,'images.csv'),'Delimiter',',','QuoteStrings',true)
        
        % Load the XML file, this we can make do without
        % info from the xml on muscle, not image level
        xml_path = fullfile(f,'results.xml');
        % if this file exists, else, try a fallback
        if exist(xml_path,'file') == 2
            s = xml2struct(fullfile(f,'results.xml'));
            q_version = classifyQumia(s);

            if q_version >= 3.0 %isfield(s,'qumia_xml')
                muscle_struct = s.qumia_xml.muscle;
                % get the correct fields
                ref_fields = xml_fields3;
            else
                muscle_struct = s.qumiaroot.metingen.muscle;
                ref_fields = xml_fields2_1;
            end

            muscles = [];
            % in few cases, only one muscle is present, this then gets
            % flattened erroneously
            if size(muscle_struct,2) == 1
                disp('Skipped ' + string(folder_name) + ', only one muscle, version ' + string(q_version));
                continue;
            end

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
            end
            if isempty(muscles)
               disp('Skipped ' + string(folder_name) + ', no muscle with score, version ' + string(q_version));
               continue; 
            end
            % one entry for each recorded muscle (and side) with scores and
            % raw values
            merged_muscles = mergeStructs(muscles{:});
            muscle_table = struct2table(merged_muscles);

            % rename older variable names
            if ~(q_version >= 3.0)
                muscle_table.Properties.VariableNames{'name_Text'} = 'Muscle';
                muscle_table.Properties.VariableNames{'EIzscore_Text'} = 'EI_zscore';
            else
                muscle_table.Properties.VariableNames{'muscle_name_Text'} = 'Muscle';
                muscle_table.Properties.VariableNames{'EI_zscore_Text'} = 'EI_zscore';
            end
            muscle_table.Properties.VariableNames{'side_Text'} = 'Side';
            muscle_table.Properties.VariableNames{'EI_Text'} = 'EI';
            % cast to number
            try
                muscle_table.EI_zscore = cellfun(@str2double,muscle_table.EI_zscore);
                muscle_table.EI = cellfun(@str2double,muscle_table.EI);
            catch
               disp('Skipped ' + string(folder_name) + ', incomplete EI information ' + string(q_version));
               continue; 
            end
           % writetable(muscle_table,fullfile(target_path,'muscles.csv'),'Delimiter',',','QuoteStrings',true)
            % join the tables to get the EI scores for each image
            image_table = outerjoin(image_table, muscle_table, 'Keys',{'Muscle','Side'}, 'Type', 'Left', 'MergeKeys', true);
        else
            disp('No xml file for: ' + string(folder_name) + ', using fallback.');
            % there's no version to load, so we just have to make a guess
            q_version = 2.1;
            % fill the image_table with NaN values for the EI scores
            image_table.EI = repelem(NaN,size(image_table,1))';
            image_table.EI_zscore = repelem(NaN,size(image_table,1))';
        end
        

        % get the patient info into a table as well
        patient_struct = anal_file.patient;
        record_info = flattenStruct2Cell(patient_struct);
        
        if q_version <= 1.66
            patient_fields = patient_fields1_66;
        else
            patient_fields = patient_fields2_1;
        end
        
        if length(record_info) ~= length(patient_fields)
           disp('Skipped ' + string(folder_name) + ', patient fields inconsistent, version ' + string(q_version));
           continue; 
        end
        record_info = cell2table(record_info, 'VariableNames', patient_fields);
        % the unique id of this record
        record_info.rid = string(inner_folder);
        % overwrite the ID because sometimes it just says FLIK in the
        % file
        record_info.pid = {id_to_copy};
        % remove personal information
        if ismember('Name', record_info.Properties.VariableNames)
             record_info.Name = [];
        end
        record_info.Birthdate = [];
        % remove the duplicate release date for consistent table size
        if ismember('ReleaseDate', record_info.Properties.VariableNames)
             record_info.ReleaseDate = [];
        end
        class_label = labeledpatients(labeledpatients.pid == record_info.pid{1},:).class;
        record_info.Class = class_label;
        
        % at least read in the first image to know the device
        dcm_path = fullfile(f,image_table.DCM{1});
        % read the metadata
        info = dicominfo(dcm_path);
        % store in the record info
        record_info.DeviceInfo = string(info.Manufacturer) + "_" + string(info.ManufacturerModelName);
        
        % store the number of images
        record_info.NumberImages = size(image_table,1);
        % store the number of unique muscles
        record_info.NumberMuscles = size(unique(muscle_table.Muscle),1);
        % store the muscles contained in the record, delimited by &
        record_info.Muscles = string(strjoin(muscle_table.Muscle, "&"));
        record_info.Sides = string(strjoin(muscle_table.Side, "&"));
        EI_scores = string(muscle_table.EI);
        EI_scores = fillmissing(EI_scores,'constant',"NaN");
        record_info.EIs = string(strjoin(EI_scores, "&"));
        
        z_scores = string(muscle_table.EI_zscore);
        z_scores = fillmissing(z_scores,'constant',"NaN");
        record_info.EIZ = string(strjoin(string(z_scores), "&"));
        
        % fill in the patient level info
        image_table.pid = repelem(string(id_to_copy),size(image_table,1))';
        % record id = folder name
        image_table.rid = repelem(string(inner_folder),size(image_table,1))';
        
        % image_table.Age = repelem(record_info.Age,size(image_table,1))';
        % image_table.Height = repelem(record_info.Height,size(image_table,1))';
        % image_table.Weight = repelem(record_info.Weight,size(image_table,1))';
        % image_table.Sex = repelem(record_info.Sex,size(image_table,1))';
        % image_table.Class = repelem(record_info.Class,size(image_table,1))';

        if analyze_images
            image_heights = cell(size(image_table,1),1);
            image_widths = cell(size(image_table,1),1);

            min_x_roi = cell(size(image_table,1),1);
            min_y_roi = cell(size(image_table,1),1);
            max_x_roi = cell(size(image_table,1),1);
            max_y_roi = cell(size(image_table,1),1);
            device_infos = cell(size(image_table,1),1);
            corrupted_images = [];
            for k=1:size(image_table,1)
                % read in the original DICOM (this throws out all
                % tabular information, including the patient name)
                dcm_path = fullfile(f,image_table.DCM{k});
                
                try
                    I = dicomread(dcm_path);
                catch
                    continue;
                end
                % ditch the two superfluous channels and save 2/3 of space
                I = I(:,:,1);

                % read the metadata as well and store some of it (TODO)
                info = dicominfo(dcm_path);
                % info.Manufacturer
                % info.ManufacturerModelName
                % info.SequenceOfUltrasoundRegions.Item_1
                device_info = string(info.Manufacturer) + "_" + string(info.ManufacturerModelName);
                device_infos{k} = device_info;
                min_x = 0;
                min_y = 0;
                % get the region info
                if isfield(info,'SequenceOfUltrasoundRegions')
                    region = info.SequenceOfUltrasoundRegions.Item_1;
                    min_x = region.RegionLocationMinX0;
                    min_y = region.RegionLocationMinY0;
                    max_x = region.RegionLocationMaxX1;
                    max_y = region.RegionLocationMaxY1;
                    % crop out only the actual ultrasound image
                    % x and y are swapped for unknown reasons
                    I = I(min_y:max_y, min_x:max_x);
                else
                    % at least set to black the first sixty pixels at the top 
                    % in order not to accidentally include any patient names
                    I(1:60,:) = 0;
                end

                is_esoate = (device_info == "ESAOTE_6100");
                % adjusted cropping behaviour for ESOATE
                % necessary as there are small marks at the bottom,
                % so no column is ever totally black
                % instead, we find the first non-black pixel for every row
                % and then take the mode of the values
                if is_esoate
                    % for each row, find the first pixel that is not black
                    offset_in = ones(size(I,1),1);
                    offset_out = ones(size(I,1),1);
                    for row_ind=1:size(I,1)
                        % find the first pixel in that row that is not black
                        not_black = I(row_ind,:) > 0;
                        in_ind = find(not_black,1,'first');
                        out_ind = find(not_black,1,'last');
                        if size(in_ind,2) == 0
                            in_ind = NaN;
                        end
                        if size(out_ind,2) == 0
                            out_ind = NaN;
                        end
                        offset_in(row_ind,:) = in_ind;
                        offset_out(row_ind,:) = out_ind;
                    end
                    offset_in = mode(offset_in);
                    offset_out = mode(offset_out);
                % default behaviour, find the biggest contiguous non-zero region
                else
                    % columns that have only zeros in them
                    zero = (sum(I) == 0);
                    zero_inds = find(zero);
                    % find the biggest gap, that is where the image is
                    [gap_size, ind] = max(diff(find(zero)));
                    offset_in = zero_inds(ind);
                    offset_out = zero_inds(ind + 1);
                end
                % crop out the black bars to the left and the right
                I = I(:,offset_in:offset_out);
                % get width and height
                image_heights{k} = size(I,1);
                image_widths{k} = size(I,2);

                if copy_images_cached
                    file_name = image_table.DCM{k}(1:end-4) + export_format;
                    new_path = char(fullfile(target_path,file_name));
                    imwrite(I,new_path);
                    % dicomwrite(I,new_path);
                end

                roi_path = fullfile(f,'roi',string(image_table.DCM{k}) + '.mat');
                min_roi = [NaN; NaN];
                max_roi = [NaN; NaN];
                r.roi = [];
                
                % some images don't have an associated ROI file
                if exist(roi_path,'file') == 2
                    r = load(roi_path);
                end
                % sometimes, it exists, but is empty
                if ~isempty(r.roi)
                    r = load(roi_path);
                    
                    % adjust the ROI values
                    r.roi(1,:) = max(1, r.roi(1,:) - single(min_x) - offset_in);
                    r.roi(2,:) = max(1, r.roi(2,:) - single(min_y));
                    min_roi = round(min(r.roi,[],2));
                    max_roi = round(max(r.roi,[],2));

                    if copy_images_cached
                        new_roi_path = fullfile(target_path,string(image_table.DCM{k}) + '.mat');    
                        save(new_roi_path,'r');
                    end
                end
                % store ROI bounding boxes
                min_x_roi{k} = min_roi(1);
                min_y_roi{k} = min_roi(2);
                max_x_roi{k} = max_roi(1);
                max_y_roi{k} = max_roi(2);
            end

            image_table.ImWidth = image_widths;
            image_table.ImHeight = image_heights;
            image_table.DeviceInfo = device_infos;
            image_table.min_h_roi = min_x_roi;
            image_table.min_w_roi = min_y_roi;
            image_table.max_h_roi = max_x_roi;
            image_table.max_w_roi = max_y_roi;
        end
        if copy_images
           format_string = export_format;
        else
           format_string = ".dcm";
        end
        % drop out images with no device info
        % this can happen in case a dcm file is corrupted
        loc=cellfun(@isempty, image_table{:,'DeviceInfo'} );
        image_table(loc,:)=[];
        image_table.Image = cellfun(@(x) x(1:end-4) + format_string, image_table.DCM);
        image_table.DCM = [];

        image_tables{counter} = image_table;
        record_infos{counter} = record_info;

        counter = counter +1;
        end

end

disp('Merging information.');

record_infos = vertcat(record_infos{:});
% writetable(patient_infos,fullfile(out_path,'patients.xlsx'))
records_file = sprintf('records_%d_to_%d.csv',start_patient,end_patient);
writetable(record_infos,fullfile(out_path,records_file),'Delimiter',',','QuoteStrings',true)

total_table = vertcat(image_tables{:});
%writetable(total_table,fullfile(out_path,'full_format_image_info.xlsx'))
image_file = sprintf('images_%d_to_%d.csv',start_patient,end_patient);

writetable(total_table,fullfile(out_path,image_file),'Delimiter',',','QuoteStrings',true)

% all_missing = [string(missing_patients); missing_xml'];
% inds = ismember(labeledpatients.pid, all_missing);
% missing_patients = labeledpatients(inds, :);
% % this messes the table up, leading 0s get deleted
% % writetable(missing_patients,fullfile(out_path,'missing_patients.xlsx'))
% writetable(missing_patients,fullfile(out_path,'missing_patients.csv'),'Delimiter',',','QuoteStrings',true)
% 
% % instead, save to a matfile, rename to allow reading in again directly
% labeledpatients = missing_patients;
% save(fullfile(out_path,'missing_patients.mat'), 'labeledpatients');
% if ~isempty(dup_results)
%     dup_results = vertcat(dup_results{:,2});
%     % writetable(dup_results,fullfile(out_path,'multi_record_patients.xlsx'))
%     writetable(dup_results,fullfile(out_path,'multi_record_patients.csv'),'Delimiter',',','QuoteStrings',true)
% end