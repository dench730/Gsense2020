import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from ccdproc import ImageFileCollection


exposures_groups = []  # <- 把你的每组10张 flat 文件列表放进来

dark_groups = (
    []
)

data_workdir = "/Volumes/Shu-H1/CCD_CMOS/ustc_cmos/cmos_flat"
flat_frames = ImageFileCollection(
    f"{data_workdir}", glob_include=f"flat*-*00*.fit", glob_exclude="*new*"
)
flat_frames.sort(["exptime", "jd"])
dark_frames = ImageFileCollection(
    f"{data_workdir}/dark", glob_include=f"dark*-*00*.fit"
)
dark_frames.sort(["exptime", "jd"])
exp_list = np.unique(flat_frames.summary["exptime"])
for exp in exp_list:
    dark_frame = dark_frames.files_filtered(include_path=True, exptime=exp)
    flat_filelist = flat_frames.files_filtered(include_path=True, exptime=exp)
    exposures_groups.append(flat_filelist)
    dark_groups.append(dark_frame)

#    ROI 格式： (y0:y1, x0:x1) ，例如中心 100x100: roi = (slice(950,1050), slice(950,1050))
# roi = None  # 或
roi = (slice(700, 1500), slice(900, 1700))

# 4) saturation threshold（避免饱和）
SATURATION_ADU = 65535  # 16-bit 相机的上限；根据你相机改


def read_fits(path):
    with fits.open(path) as h:
        return h[0].data.astype(np.float64)


def stack_mean(images):
    """逐像素计算平均图像"""
    return np.mean(images, axis=0)


def stack_var(images):
    """逐像素计算样本方差（N-1）"""
    return np.var(images, axis=0, ddof=1)


def pairwise_diff_variance(images):
    """
    两两差分法估计方差（去掉固定模式噪声更稳定）
    要求 images 中帧数是偶数，或会丢弃最后一帧。
    Var_est = 0.5 * mean((I_{2k} - I_{2k+1})^2)
    返回像素阵的方差估计（ADU^2）
    """
    n = len(images)
    m = (n // 2) * 2
    diffs = []
    for i in range(0, m, 2):
        d = images[i] - images[i + 1]
        diffs.append(d**2)
    diffs = np.stack(diffs, axis=0)
    return 0.5 * np.mean(diffs, axis=0)


def compute_gain_from_groups(
    flats_groups, dark_groups=None, roi=None, sat_threshold=SATURATION_ADU, verbose=True
):
    """
    flats_groups: list of lists (每组若干 flat 文件路径)
    dark_groups: list of lists (每组对应的 dark 文件路径) 或 None
    roi: None or (slice(y0,y1), slice(x0,x1))
    """
    mean_list = []
    var_list = []
    exposure_times = []

    for gi, flat_files in enumerate(flats_groups):
        if len(flat_files) < 2:
            raise ValueError("每组至少需要2张图（建议10张），当前组太少: ", flat_files)
        # 读取 flats
        flats = [read_fits(f) for f in flat_files]
        if roi is not None:
            flats = [f[roi] for f in flats]

        # 计算 dark 平均与 var（如果提供）
        if dark_groups and len(dark_groups) > gi and dark_groups[gi]:
            darks = [read_fits(f) for f in dark_groups[gi]]
            if roi is not None:
                darks = [d[roi] for d in darks]
            dark_mean = stack_mean(darks)
            # 用差分法或样本方差估暗场方差
            if len(darks) >= 2:
                dark_var = pairwise_diff_variance(darks)
            else:
                dark_var = stack_var(darks)
        else:
            # 如果没暗场，设为零（不推荐）
            dark_mean = 0
            dark_var = 0

        # 用两两差法估平场的像素方差（去掉 fixed pattern）
        flat_mean_img = stack_mean(flats)
        flat_var_img = pairwise_diff_variance(flats)  # ADU^2

        # 校正：减去暗的均值与方差
        corrected_mean_img = flat_mean_img - dark_mean
        corrected_var_img = flat_var_img - dark_var
        # 避免负的方差数值（数值误差），剪裁到 >=0
        corrected_var_img = np.clip(corrected_var_img, 0, None)

        # 排除饱和像素：基于 flat_mean_img 与阈值
        mask_saturated = flat_mean_img >= sat_threshold

        # 排除异常像素（hot pixels）：例如单像素方差或者均值过大
        # 这里简单排除极端值（以中位数为基准）
        mean_med = np.nanmedian(corrected_mean_img[~mask_saturated])
        var_med = np.nanmedian(corrected_var_img[~mask_saturated])
        # mask for outliers (tunable)
        mask_out = (corrected_mean_img > mean_med * 5) | (
            corrected_var_img > var_med * 20
        )
        mask_bad = mask_saturated | mask_out

        good_means = corrected_mean_img[~mask_bad].ravel()
        good_vars = corrected_var_img[~mask_bad].ravel()

        # 进一步去极端点（例如 1%-99%）
        lo_m, hi_m = np.percentile(good_means, [1, 99])
        sel = (good_means >= lo_m) & (good_means <= hi_m)
        good_means = good_means[sel]
        good_vars = good_vars[sel]

        mean_scalar = np.mean(good_means)
        var_scalar = np.mean(good_vars)

        mean_list.append(mean_scalar)
        var_list.append(var_scalar)

        # 获取曝光时间（尝试从 FITS header）
        try:
            hdr = fits.getheader(flat_files[0])
            exptime = (
                hdr.get("EXPTIME") or hdr.get("EXPOSURE") or hdr.get("EXPOS") or None
            )
            if exptime is None:
                # try parse filename for number (fallback)
                exptime = np.nan
        except Exception:
            exptime = np.nan

        exposure_times.append(exptime)

        if verbose:
            print(
                f"组 {gi}: mean={mean_scalar:.2f} ADU, var={var_scalar:.2f} ADU^2, exptime={exptime}"
            )

    # 数组化
    mean_arr = np.array(mean_list)
    var_arr = np.array(var_list)

    # 线性拟合 Var = a * Mean + b  (a = 1/gain)
    # 仅拟合线性区：剔除均值或方差为 0，或 NaN 的点
    good = np.isfinite(mean_arr) & np.isfinite(var_arr) & (mean_arr > 0)
    if np.sum(good) < 2:
        raise RuntimeError("有效点太少，无法拟合。检查数据、ROI 与饱和阈值。")

    x = mean_arr[good]
    y = var_arr[good]

    # 建议：剔除接近饱和点（均值占满比 > 0.8）
    # 已在像素层剔除了饱和像素，这里也可以手动限制最大均值
    idx_valid = x < (sat_threshold * 0.9)
    x = x[idx_valid]
    y = y[idx_valid]

    # 使用 np.polyfit 得到不确定度（带协方差）
    p, cov = np.polyfit(x, y, deg=1, cov=True)
    slope, intercept = p[0], p[1]
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    gain = 1.0 / slope
    gain_err = slope_err / (slope**2)

    # 读噪（ADU） -> 读噪 (e-) = sqrt(intercept) * gain
    read_noise_adu = np.sqrt(max(intercept, 0))
    read_noise_e = read_noise_adu * gain
    # propagate error (approx) for read noise: neglect covariance between slope/intercept for simplicity
    read_noise_e_err = (
        0.5 * (intercept_err / max(intercept, 1e-12)) ** 0.5 * gain
        if intercept > 0
        else np.nan
    )

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label=r"ADU (Variance vs Mean)")
    xs = np.linspace(np.min(x) * 0.9, np.max(x) * 1.1, 100)
    plt.plot(
        xs,
        slope * xs + intercept,
        "r-",
        label=f"fit: Var = {slope:.1f}*Mean + {intercept:.1f}",
    )
    plt.xlabel("Mean (ADU)")
    plt.ylabel(r"Variance (ADU$^2$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    results = {
        "mean_arr": mean_arr,
        "var_arr": var_arr,
        "exposure_times": exposure_times,
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "gain_e_per_adu": gain,
        "gain_err": gain_err,
        "read_noise_adu": read_noise_adu,
        "read_noise_e": read_noise_e,
        "read_noise_e_err": read_noise_e_err,
    }
    return results


if __name__ == "__main__":
    if len(exposures_groups) == 0:
        raise SystemExit(
            "请先在脚本顶部把 `exposures_groups` 填上你的每组文件列表，然后重运行脚本。"
        )
    res = compute_gain_from_groups(exposures_groups, dark_groups, roi=roi)
    print("\n=== 结果 ===")
    print(
        f"拟合斜率 a = 1/gain = {res['slope']:.6e} ± {res['slope_err']:.6e} (ADU^1→ADU^2/ADU)"
    )
    print(f"估计 gain = {res['gain_e_per_adu']:.3f} ± {res['gain_err']:.3f} e-/ADU")
    print(
        f"估计读噪 = {res['read_noise_e']:.2f} e- (≈ {res['read_noise_adu']:.2f} ADU)"
    )
    plt.show()


###
gain = res["gain_e_per_adu"] = 0.1003547171286968
gain_err = res["gain_err"] = 0.0027786215440441455
