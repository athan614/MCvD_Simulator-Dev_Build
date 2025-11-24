function rebuild_matlab_plots(varargin)
%REBUILD_MATLAB_PLOTS Recreate MCvD figures from CSV outputs using MATLAB.
%   This utility mirrors the Python analysis figure builders. It scans the
%   canonical CSV exports under results/data (SER/Nm, LoD vs distance,
%   guard/ISI, device FoM, sensitivity, capacity, etc.) and regenerates
%   publication-style figures into results/figures/matlab.
%
%   Usage:
%       rebuild_matlab_plots();                       % default paths
%       rebuild_matlab_plots('OutputDir', '...');     % custom figure root
%       rebuild_matlab_plots('DataDir', '...');       % custom CSV root
%
%   Missing datasets are skipped gracefully and reported in the console.

args = parse_inputs(varargin{:});

project_root = fileparts(fileparts(mfilename('fullpath'))); % utils -> repo root
if isempty(args.DataDir)
    args.DataDir = fullfile(project_root, 'results', 'data');
end
if isempty(args.OutputDir)
    args.OutputDir = fullfile(project_root, 'results', 'figures', 'matlab');
end
if ~isfolder(args.DataDir)
    error('Data directory not found: %s', args.DataDir);
end
if ~isfolder(args.OutputDir)
    mkdir(args.OutputDir);
end

logf('Project root: %s', project_root);
logf('Data dir    : %s', args.DataDir);
logf('Output dir  : %s', args.OutputDir);

% Core modality summaries
safe_call(@() plot_ser_vs_nm(args), 'SER vs Nm');
safe_call(@() plot_lod_vs_distance(args), 'LoD vs distance');
safe_call(@() plot_nt_pairs(args), 'CSK NT pairs');
safe_call(@() plot_csk_baselines(args), 'CSK single/dual baselines');

% Device / guard / ISI metrics
safe_call(@() plot_device_fom(args), 'Device FoM');
safe_call(@() plot_guard_frontier(args), 'Guard frontiers');
safe_call(@() plot_isi_tradeoff(args), 'ISI trade-off');
safe_call(@() plot_input_output_gain(args), 'Input/output gain');

% Hybrid benchmark / SNR sweeps
safe_call(@() plot_hybrid_benchmark(args), 'Hybrid multi-dimensional benchmark');
safe_call(@() plot_snr_vs_ts(args), 'SNR vs Ts sweeps');

% Sensitivity + organoid sweeps
safe_call(@() plot_sensitivity(args), 'Sensitivity sweeps');
safe_call(@() plot_organoid(args), 'Organoid sensitivity');

% Capacity bounds
safe_call(@() plot_capacity(args), 'Capacity bounds');

logf('Done.');
end

% -------------------------------------------------------------------------
function args = parse_inputs(varargin)
parser = inputParser;
parser.addParameter('DataDir', '', @(s) ischar(s) || isstring(s));
parser.addParameter('OutputDir', '', @(s) ischar(s) || isstring(s));
parser.parse(varargin{:});
args = parser.Results;
end

% -------------------------------------------------------------------------
function safe_call(fn, label)
try
    fn();
catch ME %#ok<NASGU>
    logf('[warn] %s skipped (%s)', label, ME.message);
end
end

% -------------------------------------------------------------------------
function plot_ser_vs_nm(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
styles = struct( ...
    'MoSK', struct('Color', [0.0, 0.45, 0.74], 'Marker', 'o'), ...
    'CSK',  struct('Color', [0.0, 0.6, 0.5],   'Marker', 's'), ...
    'Hybrid', struct('Color', [0.85, 0.33, 0.1], 'Marker', '^'));

fig = figure('Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold(ax1, 'on');

% Panel (a): SER curves per mode (CTRL states split if present)
for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('ser_vs_nm_%s', lower(mode)));
    if isempty(tbl)
        continue;
    end
    nmcol = first_column(tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if isempty(nmcol) || ~ismember('ser', tbl.Properties.VariableNames)
        continue;
    end
    use_ctrl = get_use_ctrl(tbl);
    style = styles.(mode);
    if ~isempty(use_ctrl)
        uvals = unique(use_ctrl(~isnan(use_ctrl)));
        for ui = 1:numel(uvals)
            ctrl_val = uvals(ui);
            mask = use_ctrl == ctrl_val;
            nm = as_numeric(tbl.(nmcol));
            ser = as_numeric(tbl.ser);
            nm = nm(mask); ser = ser(mask);
            [nm, ser] = clean_pair(nm, ser);
            if isempty(nm)
                continue;
            end
            ls = '-';
            if ~ctrl_val
                ls = '--';
            end
            lbl = sprintf('%s%s', mode, ternary(ctrl_val, '', ' (no CTRL)'));
            loglog(ax1, nm, ser, 'LineWidth', 1.5, 'Color', style.Color, ...
                'Marker', style.Marker, 'MarkerSize', 6, 'LineStyle', ls, ...
                'DisplayName', lbl);
        end
    else
        nm = as_numeric(tbl.(nmcol));
        ser = as_numeric(tbl.ser);
        [nm, ser] = clean_pair(nm, ser);
        if isempty(nm)
            continue;
        end
        loglog(ax1, nm, ser, 'LineWidth', 1.5, 'Color', style.Color, ...
            'Marker', style.Marker, 'MarkerSize', 6, 'DisplayName', mode);
    end
end
grid(ax1, 'on');
set(ax1, 'YScale', 'log', 'XScale', 'log');
yline(ax1, 1e-2, ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', 0.8);
xlabel(ax1, 'Nm (molecules / symbol)');
ylabel(ax1, 'SER');
title(ax1, '(a) SER vs Nm');
legend(ax1, 'Location', 'southwest');

% Panel (b): Hybrid error components
ax2 = nexttile; hold(ax2, 'on');
tbl_h = load_mode_csv(args.DataDir, 'ser_vs_nm_hybrid');
if ~isempty(tbl_h)
    nmcol = first_column(tbl_h, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if ~isempty(nmcol) && all(ismember({'ser', 'mosk_ser', 'csk_ser'}, tbl_h.Properties.VariableNames))
        nm = as_numeric(tbl_h.(nmcol));
        total = as_numeric(tbl_h.ser);
        mosk = as_numeric(tbl_h.mosk_ser);
        csk = as_numeric(tbl_h.csk_ser);
        [nm, total, mosk, csk] = clean_quad(nm, total, mosk, csk);
        loglog(ax2, nm, total, 'k-', 'LineWidth', 1.6, 'Marker', 'o', 'DisplayName', 'Total SER');
        loglog(ax2, nm, mosk, '--', 'Color', styles.MoSK.Color, 'LineWidth', 1.4, 'Marker', '^', 'DisplayName', 'MoSK errors');
        loglog(ax2, nm, csk, '-.', 'Color', styles.CSK.Color, 'LineWidth', 1.4, 'Marker', 's', 'DisplayName', 'CSK errors');
        yline(ax2, 1e-2, ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', 0.8);
        grid(ax2, 'on'); set(ax2, 'YScale', 'log', 'XScale', 'log');
        xlabel(ax2, 'Nm (molecules / symbol)');
        ylabel(ax2, 'SER');
        title(ax2, '(b) Hybrid error components');
        legend(ax2, 'Location', 'southwest');
    else
        text(0.5, 0.5, 'ser_vs_nm_hybrid.csv missing required columns', 'HorizontalAlignment', 'center', 'Parent', ax2);
        axis(ax2, 'off');
    end
else
    text(0.5, 0.5, 'No hybrid SER CSV found', 'HorizontalAlignment', 'center', 'Parent', ax2);
    axis(ax2, 'off');
end

out_path = fullfile(args.OutputDir, 'fig_ser_vs_nm_matlab.png');
save_figure(fig, out_path);
logf('[ok] SER vs Nm figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_lod_vs_distance(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
styles = {[0.0, 0.45, 0.74], [0.0, 0.6, 0.5], [0.85, 0.33, 0.1]};

fig = figure('Visible', 'off');
ax = axes(fig); hold(ax, 'on');
for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('lod_vs_distance_%s', lower(mode)));
    if isempty(tbl) || ~all(ismember({'distance_um', 'lod_nm'}, tbl.Properties.VariableNames))
        continue;
    end
    d = as_numeric(tbl.distance_um);
    lod = as_numeric(tbl.lod_nm);
    [d, lod] = clean_pair(d, lod);
    if isempty(d)
        continue;
    end
    semilogy(ax, d, lod, 'o-', 'Color', styles{k}, 'LineWidth', 1.5, ...
        'DisplayName', mode);
end
grid(ax, 'on'); set(ax, 'YScale', 'log');
xlabel(ax, 'Distance (\mum)');
ylabel(ax, 'LoD (molecules @ 1% SER)');
title(ax, 'LoD vs distance');
legend(ax, 'Location', 'northwest');
out_path = fullfile(args.OutputDir, 'fig_lod_vs_distance_matlab.png');
save_figure(fig, out_path);
logf('[ok] LoD vs distance -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_nt_pairs(args)
% Plot all CSK NT pair SER curves found in results/data/ser_vs_nm_csk_*.csv
files = dir(fullfile(args.DataDir, 'ser_vs_nm_csk_*.csv'));
if isempty(files)
    logf('... NT pair plot skipped (no ser_vs_nm_csk_*.csv)');
    return;
end

fig = figure('Visible', 'off');
ax = axes(fig); hold(ax, 'on');
colors = lines(max(8, numel(files)));
idx = 1;
for k = 1:numel(files)
    name = files(k).name;
    % Skip baseline variants handled elsewhere
    if contains(name, 'single') || contains(name, 'dual')
        continue;
    end
    tbl = load_table(fullfile(args.DataDir, name));
    nmcol = first_column(tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if isempty(tbl) || isempty(nmcol) || ~ismember('ser', tbl.Properties.VariableNames)
        continue;
    end
    nm = as_numeric(tbl.(nmcol));
    ser = as_numeric(tbl.ser);
    [nm, ser] = clean_pair(nm, ser);
    if isempty(nm)
        continue;
    end
    label = erase(name, '.csv');
    loglog(ax, nm, ser, 'LineWidth', 1.2, 'Marker', 'o', 'Color', colors(idx, :), ...
        'DisplayName', label);
    idx = idx + 1;
end
if isempty(ax.Children)
    close(fig);
    logf('... NT pair plot skipped (no usable files)');
    return;
end
grid(ax, 'on'); set(ax, 'XScale', 'log', 'YScale', 'log');
yline(ax, 1e-2, ':', 'Color', [0.2, 0.2, 0.2]);
xlabel(ax, 'Nm (molecules / symbol)');
ylabel(ax, 'SER');
title(ax, 'CSK NT pair SER sweeps');
legend(ax, 'Location', 'southwest');
out_path = fullfile(args.OutputDir, 'fig_nt_pairs_ser_matlab.png');
save_figure(fig, out_path);
logf('[ok] NT-pair SER figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_csk_baselines(args)
ser_variants = struct( ...
    'single_DA_noctrl', 'Single DA', ...
    'single_SERO_noctrl', 'Single SERO', ...
    'dual_noctrl', 'Dual (DA+SERO)');
lod_variants = struct( ...
    'single_DA_ctrl', 'Single DA (CTRL)', ...
    'single_SERO_ctrl', 'Single SERO (CTRL)', ...
    'dual_ctrl', 'Dual (CTRL)');

% SER baseline plot
fig1 = figure('Visible', 'off'); ax1 = axes(fig1); hold(ax1, 'on');
for key = fieldnames(ser_variants)'
    csv_path = fullfile(args.DataDir, ['ser_vs_nm_csk_', key{1}, '.csv']);
    if ~isfile(csv_path)
        continue;
    end
    tbl = load_table(csv_path);
    nmcol = first_column(tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if isempty(nmcol) || ~ismember('ser', tbl.Properties.VariableNames)
        continue;
    end
    nm = as_numeric(tbl.(nmcol));
    ser = as_numeric(tbl.ser);
    [nm, ser] = clean_pair(nm, ser);
    if isempty(nm)
        continue;
    end
    semilogy(ax1, nm, ser, 'o-', 'DisplayName', ser_variants.(key{1}), 'LineWidth', 1.3);
end
if ~isempty(ax1.Children)
    grid(ax1, 'on'); set(ax1, 'YScale', 'log');
    yline(ax1, 1e-2, ':', 'Color', [0.2, 0.2, 0.2]);
    xlabel(ax1, 'Nm (molecules / symbol)'); ylabel(ax1, 'SER');
    title(ax1, 'CSK baselines: SER');
    legend(ax1, 'Location', 'southwest');
    save_figure(fig1, fullfile(args.OutputDir, 'fig_csk_baseline_ser_matlab.png'));
else
    close(fig1);
end

% LoD baseline plot
fig2 = figure('Visible', 'off'); ax2 = axes(fig2); hold(ax2, 'on');
for key = fieldnames(lod_variants)'
    csv_path = fullfile(args.DataDir, ['lod_vs_distance_csk_', key{1}, '.csv']);
    if ~isfile(csv_path)
        continue;
    end
    tbl = load_table(csv_path);
    if ~all(ismember({'distance_um', 'lod_nm'}, tbl.Properties.VariableNames))
        continue;
    end
    d = as_numeric(tbl.distance_um);
    lod = as_numeric(tbl.lod_nm);
    [d, lod] = clean_pair(d, lod);
    if isempty(d)
        continue;
    end
    plot(ax2, d, lod, 's-', 'LineWidth', 1.3, 'DisplayName', lod_variants.(key{1}));
end
if ~isempty(ax2.Children)
    grid(ax2, 'on');
    xlabel(ax2, 'Distance (\mum)'); ylabel(ax2, 'LoD (Nm @ 1% SER)');
    title(ax2, 'CSK baselines: LoD');
    legend(ax2, 'Location', 'northwest');
    save_figure(fig2, fullfile(args.OutputDir, 'fig_csk_baseline_lod_matlab.png'));
else
    close(fig2);
end
end

% -------------------------------------------------------------------------
function plot_device_fom(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
fig = figure('Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold(ax1, 'on');
ax2 = nexttile; hold(ax2, 'on');

for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('device_fom_%s', lower(mode)));
    if isempty(tbl) || ~ismember('param_type', tbl.Properties.VariableNames) || ~ismember('param_value', tbl.Properties.VariableNames)
        continue;
    end
    gm_mask = strcmp(tbl.param_type, 'gm_S');
    c_mask = strcmp(tbl.param_type, 'C_tot_F');
    if ismember('delta_over_sigma_I', tbl.Properties.VariableNames)
        metric = 'delta_over_sigma_I';
    elseif ismember('delta_over_sigma_Q', tbl.Properties.VariableNames)
        metric = 'delta_over_sigma_Q';
    else
        continue;
    end
    if any(gm_mask)
        gm = as_numeric(tbl.param_value(gm_mask)) * 1e3; % S -> mS
        val = as_numeric(tbl.(metric)(gm_mask));
        [gm, val] = clean_pair(gm, val);
        if ~isempty(gm)
            plot(ax1, gm, val, 'o-', 'LineWidth', 1.3, 'DisplayName', mode);
        end
    end
    if any(c_mask)
        c = as_numeric(tbl.param_value(c_mask)) * 1e9; % F -> nF
        val = as_numeric(tbl.(metric)(c_mask));
        [c, val] = clean_pair(c, val);
        if ~isempty(c)
            plot(ax2, c, val, 'o-', 'LineWidth', 1.3, 'DisplayName', mode);
        end
    end
end

format_subplot(ax1, 'g_m (mS)', 'Delta/sigma (device)', 'Device FoM vs g_m');
format_subplot(ax2, 'C_{tot} (nF)', 'Delta/sigma (device)', 'Device FoM vs C_{tot}');
out_path = fullfile(args.OutputDir, 'fig_device_fom_matlab.png');
save_figure(fig, out_path);
logf('[ok] Device FoM figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_guard_frontier(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
fig = figure('Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold(ax1, 'on');
ax2 = nexttile; hold(ax2, 'on');

for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('guard_frontier_%s', lower(mode)));
    if isempty(tbl) || ~all(ismember({'distance_um', 'best_guard_factor', 'max_irt_bps'}, tbl.Properties.VariableNames))
        continue;
    end
    d = as_numeric(tbl.distance_um);
    gf = as_numeric(tbl.best_guard_factor);
    irt = as_numeric(tbl.max_irt_bps);
    [d, gf, irt] = clean_triple(d, gf, irt);
    if isempty(d)
        continue;
    end
    plot(ax1, d, gf, 'o-', 'LineWidth', 1.3, 'DisplayName', mode);
    plot(ax2, d, irt, 'o-', 'LineWidth', 1.3, 'DisplayName', mode);
end
format_subplot(ax1, 'Distance (\mum)', 'Best guard factor (fraction Ts)', 'Guard-factor frontier');
ylim(ax1, [0, 1.05]);
format_subplot(ax2, 'Distance (\mum)', 'Max IRT (bits/s)', 'ISI-robust throughput');

out_path = fullfile(args.OutputDir, 'fig_guard_frontier_matlab.png');
save_figure(fig, out_path);
logf('[ok] Guard frontier figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_isi_tradeoff(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
fig = figure('Visible', 'off');
ax = axes(fig); hold(ax, 'on');
for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('isi_tradeoff_%s', lower(mode)));
    if isempty(tbl)
        continue;
    end
    gfcol = first_column(tbl, {'guard_factor', 'pipeline.guard_factor'});
    if isempty(gfcol) || ~ismember('ser', tbl.Properties.VariableNames)
        continue;
    end
    gf = as_numeric(tbl.(gfcol));
    ser = as_numeric(tbl.ser);
    use_ctrl = get_use_ctrl(tbl);
    if ~isempty(use_ctrl)
        uvals = unique(use_ctrl(~isnan(use_ctrl)));
    else
        uvals = [];
    end
    if isempty(uvals)
        [gf, ser] = clean_pair(gf, ser);
        if isempty(gf)
            continue;
        end
        semilogy(ax, gf, ser, 'o-', 'LineWidth', 1.3, 'DisplayName', mode);
    else
        for ui = 1:numel(uvals)
            ctrl_val = uvals(ui);
            mask = use_ctrl == ctrl_val;
            g = gf(mask); s = ser(mask);
            [g, s] = clean_pair(g, s);
            if isempty(g)
                continue;
            end
            lbl = sprintf('%s - %s', mode, ternary(ctrl_val, 'CTRL', 'no CTRL'));
            ls = ternary(ctrl_val, '-', '--');
            semilogy(ax, g, s, 'o', 'LineStyle', ls, 'LineWidth', 1.3, 'DisplayName', lbl);
        end
    end
end
grid(ax, 'on'); set(ax, 'YScale', 'log');
xlabel(ax, 'Guard factor (fraction Ts)'); ylabel(ax, 'SER');
title(ax, 'ISI robustness');
yline(ax, 1e-2, ':', 'Color', [0.2, 0.2, 0.2]);
legend(ax, 'Location', 'northwest');
out_path = fullfile(args.OutputDir, 'fig_isi_tradeoff_matlab.png');
save_figure(fig, out_path);
logf('[ok] ISI trade-off figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_input_output_gain(args)
modes = {'MoSK', 'CSK', 'Hybrid'};
fig = figure('Visible', 'off');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold(ax1, 'on');
ax2 = nexttile; hold(ax2, 'on');

for k = 1:numel(modes)
    mode = modes{k};
    tbl = load_mode_csv(args.DataDir, sprintf('ser_vs_nm_%s', lower(mode)));
    if isempty(tbl)
        continue;
    end
    nmcol = first_column(tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    q_col = first_column(tbl, {'delta_Q_diff', 'delta_over_sigma_Q'});
    i_col = first_column(tbl, {'delta_I_diff', 'delta_over_sigma_I'});
    if isempty(nmcol)
        continue;
    end
    if ~isempty(q_col)
        nm = as_numeric(tbl.(nmcol));
        val = as_numeric(tbl.(q_col));
        [nm, val] = clean_pair(nm, val);
        if ~isempty(nm)
            scatter(ax1, nm, val, 24, 'filled', 'DisplayName', mode);
        end
    end
    if ~isempty(i_col)
        nm = as_numeric(tbl.(nmcol));
        val = as_numeric(tbl.(i_col));
        [nm, val] = clean_pair(nm, val);
        if ~isempty(nm)
            scatter(ax2, nm, val, 24, 'filled', 'DisplayName', mode);
        end
    end
end
set(ax1, 'XScale', 'log'); grid(ax1, 'on');
set(ax2, 'XScale', 'log'); grid(ax2, 'on');
xlabel(ax1, 'Nm (molecules / symbol)'); ylabel(ax1, 'DeltaQ or DeltaQ/sigma');
title(ax1, 'Charge-domain gain vs Nm');
xlabel(ax2, 'Nm (molecules / symbol)'); ylabel(ax2, 'DeltaI or DeltaI/sigma');
title(ax2, 'Current-domain gain vs Nm');
legend(ax1, 'Location', 'best');
legend(ax2, 'Location', 'best');
out_path = fullfile(args.OutputDir, 'fig_input_output_gain_matlab.png');
save_figure(fig, out_path);
logf('[ok] Input-output gain -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_hybrid_benchmark(args)
ser_tbl = load_mode_csv(args.DataDir, 'ser_vs_nm_hybrid');
lod_tbl = load_mode_csv(args.DataDir, 'lod_vs_distance_hybrid');
isi_tbl = load_mode_csv(args.DataDir, 'isi_tradeoff_hybrid');
hds_grid = load_table_if_exists(fullfile(args.DataDir, 'hybrid_hds_grid.csv'));
isi_grid = load_table_if_exists(fullfile(args.DataDir, 'isi_grid_hybrid.csv'));

fig = figure('Visible', 'off');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel A: HDS surface or curves
axA = nexttile; hold(axA, 'on');
nmcol = '';
if ~isempty(hds_grid)
    nmcol = first_column(hds_grid, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
end
if ~isempty(hds_grid) && ~isempty(nmcol) && ismember('distance_um', hds_grid.Properties.VariableNames) ...
        && ismember('ser', hds_grid.Properties.VariableNames)
    g = pivot(hds_grid, 'distance_um', nmcol, 'ser');
    if ~isempty(g)
        imagesc(axA, g.X, g.Y, g.Z);
        set(axA, 'YDir', 'normal', 'XScale', 'log');
        colorbar(axA);
        xlabel(axA, 'Nm (log)'); ylabel(axA, 'Distance (\mum)');
        title(axA, '(A) Hybrid decision surface (SER)');
    end
elseif ~isempty(ser_tbl)
    nmcol = first_column(ser_tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if ~isempty(nmcol) && all(ismember({'ser', 'mosk_ser', 'csk_ser'}, ser_tbl.Properties.VariableNames))
        nm = as_numeric(ser_tbl.(nmcol));
        [nm, total, mosk, csk] = clean_quad(nm, as_numeric(ser_tbl.ser), ...
            as_numeric(ser_tbl.mosk_ser), as_numeric(ser_tbl.csk_ser));
        loglog(axA, nm, total, 'k-', 'LineWidth', 1.4, 'DisplayName', 'Total SER');
        loglog(axA, nm, mosk, '--', 'Color', [0.8, 0.3, 0.3], 'LineWidth', 1.2, 'DisplayName', 'MoSK');
        loglog(axA, nm, csk, '-.', 'Color', [0.3, 0.4, 0.8], 'LineWidth', 1.2, 'DisplayName', 'CSK');
        grid(axA, 'on'); set(axA, 'XScale', 'log', 'YScale', 'log');
        yline(axA, 1e-2, ':', 'Color', [0.2, 0.2, 0.2]);
        xlabel(axA, 'Nm'); ylabel(axA, 'SER'); legend(axA, 'Location', 'southwest');
        title(axA, '(A) Hybrid decision components');
    else
        text(0.5, 0.5, 'No hybrid SER columns', 'HorizontalAlignment', 'center', 'Parent', axA);
        axis(axA, 'off');
    end
else
    text(0.5, 0.5, 'No hybrid SER CSV found', 'HorizontalAlignment', 'center', 'Parent', axA);
    axis(axA, 'off');
end

% Panel B: LoD vs distance (hybrid)
axB = nexttile; hold(axB, 'on');
if ~isempty(lod_tbl) && all(ismember({'distance_um', 'lod_nm'}, lod_tbl.Properties.VariableNames))
    d = as_numeric(lod_tbl.distance_um);
    lod = as_numeric(lod_tbl.lod_nm);
    [d, lod] = clean_pair(d, lod);
    if ~isempty(d)
        semilogy(axB, d, lod, 'o-', 'LineWidth', 1.3);
        grid(axB, 'on'); ylabel(axB, 'LoD (Nm @ 1% SER)'); xlabel(axB, 'Distance (\mum)');
        title(axB, '(B) LoD vs distance (Hybrid)');
    end
else
    text(0.5, 0.5, 'No lod_vs_distance_hybrid.csv', 'HorizontalAlignment', 'center', 'Parent', axB);
    axis(axB, 'off');
end

% Panel C: ISI grid / trade-off throughput
axC = nexttile; hold(axC, 'on');
if ~isempty(isi_grid) && all(ismember({'guard_factor', 'distance_um', 'symbol_period_s', 'ser'}, isi_grid.Properties.VariableNames))
    isi_grid.R_eff_bps = 2 ./ as_numeric(isi_grid.symbol_period_s) .* (1 - as_numeric(isi_grid.ser));
    g = pivot(isi_grid, 'distance_um', 'guard_factor', 'R_eff_bps');
    if ~isempty(g)
        imagesc(axC, g.X, g.Y, g.Z);
        set(axC, 'YDir', 'normal');
        colorbar(axC);
        xlabel(axC, 'Guard factor'); ylabel(axC, 'Distance (\mum)');
        title(axC, '(C) R_{eff} heatmap');
    end
elseif ~isempty(isi_tbl)
    gfcol = first_column(isi_tbl, {'guard_factor', 'pipeline.guard_factor'});
    if ~isempty(gfcol) && all(ismember({'symbol_period_s', 'ser'}, isi_tbl.Properties.VariableNames))
        gf = as_numeric(isi_tbl.(gfcol));
        Ts = as_numeric(isi_tbl.symbol_period_s);
        ser = as_numeric(isi_tbl.ser);
        R = 2 ./ Ts .* (1 - ser);
        [gf, R] = clean_pair(gf, R);
        plot(axC, gf, R, 'o-', 'LineWidth', 1.3);
        grid(axC, 'on'); xlabel(axC, 'Guard factor'); ylabel(axC, 'R_{eff} (bits/s)');
        title(axC, '(C) ISI-robust throughput');
    end
else
    text(0.5, 0.5, 'No ISI trade-off data', 'HorizontalAlignment', 'center', 'Parent', axC);
    axis(axC, 'off');
end

% Panel D: Hybrid SER overview
axD = nexttile; hold(axD, 'on');
if ~isempty(ser_tbl)
    nmcol = first_column(ser_tbl, {'pipeline_Nm_per_symbol', 'pipeline.Nm_per_symbol', 'Nm_per_symbol', 'Nm'});
    if ~isempty(nmcol) && ismember('ser', ser_tbl.Properties.VariableNames)
        nm = as_numeric(ser_tbl.(nmcol));
        ser = as_numeric(ser_tbl.ser);
        [nm, ser] = clean_pair(nm, ser);
        semilogy(axD, nm, ser, 'o-', 'LineWidth', 1.3);
        yline(axD, 1e-2, ':', 'Color', [0.2, 0.2, 0.2]);
        grid(axD, 'on'); set(axD, 'XScale', 'log');
        xlabel(axD, 'Nm'); ylabel(axD, 'SER');
        title(axD, '(D) Hybrid SER (overview)');
    else
        axis(axD, 'off');
    end
else
    axis(axD, 'off');
end

out_path = fullfile(args.OutputDir, 'fig_hybrid_matlab.png');
save_figure(fig, out_path);
logf('[ok] Hybrid benchmark figure -> %s', out_path);
end

% -------------------------------------------------------------------------
function plot_snr_vs_ts(args)
files = dir(fullfile(args.DataDir, 'snr_vs_ts_*.csv'));
if isempty(files)
    logf('... No snr_vs_ts_* CSVs found');
    return;
end
for k = 1:numel(files)
    tbl = load_table(fullfile(args.DataDir, files(k).name));
    if isempty(tbl)
        continue;
    end
    Ts_col = first_column(tbl, {'symbol_period_s', 'pipeline.symbol_period_s', 'Ts'});
    snr_col = first_column(tbl, {'snr_db', 'snr_plot', 'snr_i_db', 'snr_q_db'});
    if isempty(Ts_col) || isempty(snr_col)
        continue;
    end
    Ts = as_numeric(tbl.(Ts_col));
    snr = as_numeric(tbl.(snr_col));
    [Ts, snr] = clean_pair(Ts, snr);
    if isempty(Ts)
        continue;
    end
    fig = figure('Visible', 'off');
    ax = axes(fig); hold(ax, 'on');
    plot(ax, Ts, snr, 'o-', 'LineWidth', 1.3);
    grid(ax, 'on'); xlabel(ax, 'Symbol period T_s (s)'); ylabel(ax, 'SNR (dB)');
    title(ax, sprintf('SNR vs Ts (%s)', erase(files(k).name, '.csv')));
    out_path = fullfile(args.OutputDir, [erase(files(k).name, '.csv'), '_matlab.png']);
    save_figure(fig, out_path);
    close(fig);
    logf('[ok] SNR vs Ts -> %s', out_path);
end
end

% -------------------------------------------------------------------------
function plot_sensitivity(args)
files = dir(fullfile(args.DataDir, 'sensitivity_*.csv'));
if isempty(files)
    logf('... Sensitivity plots skipped (no sensitivity_*.csv)');
    return;
end
metric_cols = {'lod_nm', 'ser', 'ser_eval', 'snr_db', 'snr_q_db', 'snr_i_db', ...
    'delta_over_sigma_Q', 'delta_over_sigma_I', 'delta_Q_diff', 'delta_I_diff'};
for k = 1:numel(files)
    tbl = load_table(fullfile(args.DataDir, files(k).name));
    if isempty(tbl)
        continue;
    end
    xcol = first_numeric_except(tbl, metric_cols);
    ycol = first_column(tbl, metric_cols);
    if isempty(xcol) || isempty(ycol)
        continue;
    end
    x = as_numeric(tbl.(xcol));
    y = as_numeric(tbl.(ycol));
    [x, y] = clean_pair(x, y);
    if isempty(x)
        continue;
    end
    fig = figure('Visible', 'off');
    ax = axes(fig); hold(ax, 'on');
    plot(ax, x, y, 'o-', 'LineWidth', 1.3);
    grid(ax, 'on');
    xlabel(ax, sanitize_label(xcol)); ylabel(ax, sanitize_label(ycol));
    title(ax, sprintf('%s', erase(files(k).name, '.csv')));
    out_path = fullfile(args.OutputDir, [erase(files(k).name, '.csv'), '_matlab.png']);
    save_figure(fig, out_path);
    close(fig);
    logf('[ok] Sensitivity figure -> %s', out_path);
end
end

% -------------------------------------------------------------------------
function plot_organoid(args)
files = dir(fullfile(args.DataDir, 'organoid_*_sensitivity*.csv'));
if isempty(files)
    logf('... Organoid plots skipped (no organoid_* CSVs)');
    return;
end
metric_cols = {'lod_nm', 'ser', 'snr_db', 'snr_q_db', 'snr_i_db', 'psd'};
for k = 1:numel(files)
    tbl = load_table(fullfile(args.DataDir, files(k).name));
    if isempty(tbl)
        continue;
    end
    xcol = first_numeric_except(tbl, metric_cols);
    ycol = first_column(tbl, metric_cols);
    if isempty(xcol) || isempty(ycol)
        continue;
    end
    x = as_numeric(tbl.(xcol));
    y = as_numeric(tbl.(ycol));
    [x, y] = clean_pair(x, y);
    if isempty(x)
        continue;
    end
    fig = figure('Visible', 'off');
    ax = axes(fig); hold(ax, 'on');
    plot(ax, x, y, 'o-', 'LineWidth', 1.3);
    grid(ax, 'on');
    xlabel(ax, sanitize_label(xcol)); ylabel(ax, sanitize_label(ycol));
    title(ax, sprintf('%s', erase(files(k).name, '.csv')));
    out_path = fullfile(args.OutputDir, [erase(files(k).name, '.csv'), '_matlab.png']);
    save_figure(fig, out_path);
    close(fig);
    logf('[ok] Organoid figure -> %s', out_path);
end
end

% -------------------------------------------------------------------------
function plot_capacity(args)
files = dir(fullfile(args.DataDir, 'capacity_bounds*.csv'));
if isempty(files)
    logf('... Capacity plot skipped (no capacity_bounds*.csv)');
    return;
end
tbl = load_table(fullfile(args.DataDir, files(1).name));
required_cols = {'mode', 'distance_um', 'I_soft_bits', 'I_hd_bits', 'I_sym_ceiling_bits'};
if isempty(tbl) || ~all(ismember(required_cols, tbl.Properties.VariableNames))
    logf('... Capacity plot skipped (missing required columns)');
    return;
end
fig = figure('Visible', 'off');
ax = axes(fig); hold(ax, 'on');
modes = unique(tbl.mode, 'stable');
for k = 1:numel(modes)
    mask = strcmp(tbl.mode, modes{k});
    d = as_numeric(tbl.distance_um(mask));
    soft = as_numeric(tbl.I_soft_bits(mask));
    [d, soft] = clean_pair(d, soft);
    if isempty(d)
        continue;
    end
    plot(ax, d, soft, 'o-', 'LineWidth', 1.3, 'DisplayName', sprintf('%s (soft)', modes{k}));
    if ismember('I_hd_bits', tbl.Properties.VariableNames)
        hard = as_numeric(tbl.I_hd_bits(mask));
        [d2, hard] = clean_pair(as_numeric(tbl.distance_um(mask)), hard);
        if ~isempty(d2)
            plot(ax, d2, hard, '--', 'LineWidth', 1.1, 'DisplayName', sprintf('%s (hard)', modes{k}));
        end
    end
end
grid(ax, 'on'); xlabel(ax, 'Distance (\mum)'); ylabel(ax, 'MI (bits)');
title(ax, 'Capacity bounds vs distance');
legend(ax, 'Location', 'best');
out_path = fullfile(args.OutputDir, 'fig_capacity_matlab.png');
save_figure(fig, out_path);
logf('[ok] Capacity figure -> %s', out_path);
end

% ============================ Helper utilities ============================
function tbl = load_mode_csv(data_dir, base)
% Prefer canonical name; fall back to first wildcard match.
canonical = fullfile(data_dir, [base, '.csv']);
if isfile(canonical)
    tbl = load_table(canonical);
    return;
end
files = dir(fullfile(data_dir, [base, '*.csv']));
if isempty(files)
    tbl = table();
    return;
end
tbl = load_table(fullfile(data_dir, files(1).name));
end

function tbl = load_table_if_exists(path)
if isfile(path)
    tbl = load_table(path);
else
    tbl = table();
end
end

function tbl = load_table(path)
try
    opts = detectImportOptions(path, 'NumHeaderLines', 0);
    tbl = readtable(path, opts);
catch
    tbl = table();
end
end

function col = first_column(tbl, candidates)
col = '';
for k = 1:numel(candidates)
    if ismember(candidates{k}, tbl.Properties.VariableNames)
        col = candidates{k};
        return;
    end
end
end

function col = first_numeric_except(tbl, blacklist)
col = '';
for k = 1:numel(tbl.Properties.VariableNames)
    name = tbl.Properties.VariableNames{k};
    if ismember(name, blacklist)
        continue;
    end
    vals = tbl.(name);
    if isnumeric(vals) || islogical(vals)
        col = name; return;
    elseif iscell(vals)
        try
            probe = cellfun(@str2double, vals);
            if any(isfinite(probe))
                col = name; return;
            end
        catch
        end
    end
end
end

function v = as_numeric(x)
if isnumeric(x)
    v = double(x);
elseif iscell(x)
    v = nan(size(x));
    for k = 1:numel(x)
        v(k) = str2double(string(x{k}));
    end
else
    v = double(x);
end
end

function ctrl = get_use_ctrl(tbl)
if ismember('use_ctrl', tbl.Properties.VariableNames)
    ctrl = as_numeric(tbl.use_ctrl);
else
    ctrl = [];
end
end

function [x, y] = clean_pair(x, y)
mask = isfinite(x) & isfinite(y);
x = x(mask); y = y(mask);
[x, order] = sort(x);
y = y(order);
end

function [x, y, z] = clean_triple(x, y, z)
mask = isfinite(x) & isfinite(y) & isfinite(z);
x = x(mask); y = y(mask); z = z(mask);
[x, order] = sort(x);
y = y(order); z = z(order);
end

function [a, b, c, d] = clean_quad(a, b, c, d)
mask = isfinite(a) & isfinite(b) & isfinite(c) & isfinite(d);
a = a(mask); b = b(mask); c = c(mask); d = d(mask);
[a, order] = sort(a);
b = b(order); c = c(order); d = d(order);
end

function out = ternary(cond, trueVal, falseVal)
if cond
    out = trueVal;
else
    out = falseVal;
end
end

function save_figure(fig, path)
if ~isfolder(fileparts(path))
    mkdir(fileparts(path));
end
exportgraphics(fig, path, 'Resolution', 300);
close(fig);
end

function format_subplot(ax, xlabel_txt, ylabel_txt, title_txt)
grid(ax, 'on');
xlabel(ax, xlabel_txt);
ylabel(ax, ylabel_txt);
title(ax, title_txt);
legend(ax, 'Location', 'best');
end

function lbl = sanitize_label(name)
lbl = strrep(name, '_', ' ');
end

function g = pivot(tbl, row_col, col_col, val_col)
try
    rows = unique(as_numeric(tbl.(row_col)));
    cols = unique(as_numeric(tbl.(col_col)));
    rows = rows(isfinite(rows));
    cols = cols(isfinite(cols));
    Z = nan(numel(rows), numel(cols));
    for i = 1:numel(rows)
        for j = 1:numel(cols)
            mask = isfinite(as_numeric(tbl.(row_col))) & isfinite(as_numeric(tbl.(col_col))) ...
                & as_numeric(tbl.(row_col)) == rows(i) & as_numeric(tbl.(col_col)) == cols(j);
            vals = as_numeric(tbl.(val_col));
            vals = vals(mask);
            if ~isempty(vals)
                Z(i, j) = median(vals, 'omitnan');
            end
        end
    end
    g = struct('X', cols, 'Y', rows, 'Z', Z);
catch
    g = [];
end
end

function logf(fmt, varargin)
fprintf('%s\n', sprintf(fmt, varargin{:}));
end
