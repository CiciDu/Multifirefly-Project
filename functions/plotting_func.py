import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from matplotlib import cm



"""### **PlotTrials** (for LSTM)

Among other things, here I eliminate the condition "ff_distance" < 250
"""


def PlotTrials(cum_mx, cum_my, cum_speed, cum_speeddummy, R, ff_position_during_this_trial, ff_real_position_sorted,
               ff_believed_position_sorted, cluster_dataframe_point, ff_dataframe, ffs_around_target_positions,
               fig, axes,
               currentTrial,
               num_trials,
               player="monkey",
               trail_color="orange",  # "orange" or "viridis" or None
               show_reward_boundary=False,
               show_stops=False,
               show_colorbar=False,
               show_believed_target_positions=False,
               show_connect_path_target=np.array([]),  # np.array([]) or target_nums
               show_connect_path_pre_target=np.array([]),  # np.array([]) or target_nums
               show_connect_path_ff=np.array([]),
               trial_to_show_cluster=None,  # None, 0, or -1
               show_scale_bar=False,
               trial_to_show_cluster_around_target=None,
               cluster_on_off_lines=False,
               show_start=True,
               zoom_in = False,
               # target_nums = "default",
               ):
    print(f"Trial {currentTrial}")
    if player == "RL/LSTM":
        cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
        if trail_color == "orange":
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=70, color="orange", zorder=2)
        elif trail_color == "viridis":
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=70, c=cum_speed, zorder=2)
        else:
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=70, color="yellow", zorder=2)

        if show_start:
            # Plot the start
            axes.scatter(cum_mxy_rotate[0, 0], cum_mxy_rotate[1, 0], marker='^', s=220, color="gold", zorder=3,
                         alpha=0.5)

        if show_stops:
            zerospeed_index = np.where(cum_speeddummy == 0)
            zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
            zerospeed_rotate = np.matmul(R, np.stack((zerospeedx, zerospeedy)))
            axes.scatter(zerospeed_rotate[0], zerospeed_rotate[1], marker='*', s=160, alpha=0.7, color="black",
                         zorder=2)

        ff_position_rotate = np.matmul(R, np.stack(
            (ff_position_during_this_trial.T[0], ff_position_during_this_trial.T[1])))
        axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=10, color="magenta", zorder=2)

        if show_believed_target_positions:
            ff_believed_position_rotate = np.matmul(R, np.stack((ff_believed_position_sorted[
                                                                 currentTrial - num_trials + 1:currentTrial + 1].T[0],
                                                                 ff_believed_position_sorted[
                                                                 currentTrial - num_trials + 1:currentTrial + 1].T[1])))
            axes.scatter(ff_believed_position_rotate[0], ff_believed_position_rotate[1], marker='*', s=185, color="red",
                         alpha=0.75, zorder=2)

        if show_reward_boundary:
            for i in ff_position_rotate.T:
                circle2 = plt.Circle((i[0], i[1]), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
                axes.add_patch(circle2)

        if trial_to_show_cluster != None:
            # Find the indices of ffs in the cluster
            cluster_indices = np.unique(cluster_dataframe_point[cluster_dataframe_point[
                                                                    'target_index'] == currentTrial + trial_to_show_cluster].ff_index.to_numpy())
            cluster_ff_positions = ff_real_position_sorted[np.array(cluster_indices)]
            cluster_ff_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
            axes.scatter(cluster_ff_rotate[0], cluster_ff_rotate[1], marker='o', c="blue", s=25, zorder=4)

        if show_connect_path_target.any():
            xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_target)]
            xy_onoff_lines = np.array(
                xy_onoff_lines.loc[(xy_onoff_lines['ff_index'] == currentTrial) & (xy_onoff_lines['visible'] == 1)][
                    ['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=50, c="green", alpha=0.8, zorder=5)

        if show_connect_path_pre_target.any():
            xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_pre_target)]
            xy_onoff_lines = np.array(
                xy_onoff_lines.loc[(xy_onoff_lines['ff_index'] == currentTrial - 1) & (xy_onoff_lines['visible'] == 1)][
                    ['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=65, c="aqua", alpha=0.8, zorder=3)

        if show_connect_path_ff.any():
            temp_dataframe = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_ff)]
            temp_dataframe = temp_dataframe.loc[(temp_dataframe['visible'] == 1)][
                ['ff_x', 'ff_y', 'monkey_x', 'monkey_y']]
            # temp_dataframe = temp_dataframe.loc[~temp_dataframe['ff_index'].isin(target_nums)]
            temp_array = temp_dataframe.to_numpy()
            temp_ff_positions = np.matmul(R, temp_array[:, :2].T)
            temp_monkey_positions = np.matmul(R, temp_array[:, 2:].T)
            for j in range(len(temp_array)):
                axes.plot(np.stack([temp_ff_positions[0, j], temp_monkey_positions[0, j]]),
                          np.stack([temp_ff_positions[1, j], temp_monkey_positions[1, j]]), '-', alpha=0.3,
                          linewidth=1.5, c="#a940f5")
                # axes.plot(temp_ff_positions[0,j], temp_ff_positions[1,j], '-', alpha=0.2, marker="o", markersize=5, color="brown")

        # if show_connect_path_ff.any():
        #   if num_trials == 1:
        #     x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 400, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
        #   else:
        #     target_nums = np.arange(currentTrial-num_trials+1, currentTrial+1)
        #     x, y = connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, 400, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
        #   for j in range(len(x)):
        #     xy_rotate = np.matmul(R, np.stack((x[j], y[j])))
        #     axes.plot(xy_rotate[0], xy_rotate[1], '-', alpha=0.2, c="#a940f5")
        #     axes.plot(xy_rotate[0][0], xy_rotate[1][0], '-', alpha=0.2, marker="o", markersize=5, color="brown")

        if trial_to_show_cluster_around_target != None:
            cluster_ff_pos = ffs_around_target_positions[currentTrial + trial_to_show_cluster_around_target]
            if len(cluster_ff_pos) > 0:
                ffs_around_target_rotate = np.matmul(R, np.stack((cluster_ff_pos.T[0], cluster_ff_pos.T[1])))
                axes.scatter(ffs_around_target_rotate[0], ffs_around_target_rotate[1], marker='o', s=30, color="blue",
                             zorder=4)
            if cluster_on_off_lines:
                # Find on_off_lines for ffs in the cluster
                for i in range(len(cluster_ff_pos)):
                    index = np.array(ff_dataframe[(np.isclose(np.array(ff_dataframe['ff_x']), cluster_ff_pos[i, 0])) & (
                        np.isclose(np.array(ff_dataframe['ff_y']), cluster_ff_pos[i, 1]))]['ff_index'])
                    if len(index) > 0:
                        index = index[0]
                        # index = ffs_around_target_indices[currentTrial-trial_to_show_cluster_around_target][i]
                        xy_onoff_lines = ff_dataframe.loc[
                            (ff_dataframe['time'] >= duration[0]) & (ff_dataframe['time'] <= duration[1])]
                        xy_onoff_lines = xy_onoff_lines.loc[
                            (xy_onoff_lines['ff_index'] == index) & (xy_onoff_lines['visible'] == 1)]
                        xy_onoff_lines2 = np.array(xy_onoff_lines[['monkey_x', 'monkey_y']])
                        onoff_lines_rotate = np.matmul(R, xy_onoff_lines2.T)
                        axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=80 - 10 * i,
                                     color=list_of_colors[i], alpha=0.8, zorder=3 + i)
                        # Use corresponding color for that ff
                        xy_onoff_lines3 = np.array(xy_onoff_lines[['ff_x', 'ff_y']])
                        ffs_around_target_rotate = np.matmul(R, xy_onoff_lines3.T)
                        axes.scatter(ffs_around_target_rotate[0], ffs_around_target_rotate[1], marker='o', s=140,
                                     alpha=0.8, color=list_of_colors[i], zorder=3)

        if show_scale_bar:
            scale1 = ScaleBar(
                dx=1, length_fraction=0.2, fixed_value=100,
                location='upper left',  # in relation to the whole plot
                label_loc='left', scale_loc='bottom'  # in relation to the line
            )
            axes.add_artist(scale1)
            axes.xaxis.set_major_locator(mtick.NullLocator())
            axes.yaxis.set_major_locator(mtick.NullLocator())

        xmin, xmax = np.min(cum_mxy_rotate[0]), np.max(cum_mxy_rotate[0])
        ymin, ymax = np.min(cum_mxy_rotate[1]), np.max(cum_mxy_rotate[1])
        bigger_width = max(xmax - xmin, ymax - ymin)
        xmiddle, ymiddle = (xmin + xmax) / 2, (ymin + ymax) / 2
        xmin, xmax = xmiddle - bigger_width / 2, xmiddle + bigger_width / 2
        ymin, ymax = ymiddle - bigger_width / 2, ymiddle + bigger_width / 2
        margin = max(bigger_width / 5, 150)
        axes.set_xlim((xmin - margin, xmax + margin))
        axes.set_ylim((ymin - margin, ymax + margin))
        axes.set_aspect('equal')

        if show_colorbar == True:
            # Make the black and red colorbar
            A = np.reshape([1, 2, 1, 2, 2, 2], (2, 3))  # The numbers don't matter much
            norm_bins = np.array([0.5, 1.5, 2.5])
            # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
            col_dict = {1: "black", 2: "red"}
            # We create a colormar from our list of colors
            speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
            ## Make normalizer and formatter
            norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
            labels = np.array(["No Reward", "Reward"])
            fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
            # Plot our figure
            im = axes.imshow(A, cmap=speed_cm, extent=[0, 0, 0, 0], norm=norm)
            cax2 = fig.add_axes([0.95, 0.15, 0.05, 0.2])
            cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
            cb.ax.tick_params(width=0)
            cb.ax.set_title('Stopping Points', ha='left')
            if trail_color == "orange":
                A = np.reshape([1, 2, 1, 2, 2, 2], (2, 3))  # The numbers don't matter much
                norm_bins = np.array([0.5, 1.5, 2.5])
                # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
                col_dict = {1: "green", 2: "orange"}
                # We create a colormar from our list of colors
                speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
                ## Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
                labels = np.array(["Top Target Visible", "Top Target Not Visible"])
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
                # Plot our figure
                im = axes.imshow(A, cmap=speed_cm, extent=[0, 0, 0, 0], norm=norm)
                cax2 = fig.add_axes([0.95, 0.5, 0.05, 0.2])
                cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
                cb.ax.tick_params(width=0)
                cb.ax.set_title('Path Colors', ha='left', y=1.04)
            elif trail_color == "viridis":
                cmap = cm.viridis
                norm = matplotlib.colors.Normalize(vmin=0, vmax=200)
                cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
                cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                    cax=cax, orientation='vertical')
                cbar.ax.set_title('Speed(cm/s)', ha='left', y=1.04)
                cbar.ax.tick_params(axis='y', color='white', direction="in", right=True, length=5, width=1.5)
                cbar.outline.remove()
    elif player == "monkey":
        cum_mxy_rotate = np.matmul(R, np.stack((cum_mx, cum_my)))
        if trail_color == "orange":
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=10, color="orange", zorder=2)
        elif trail_color == "viridis":
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=10, c=cum_speed, zorder=2)
        else:
            axes.scatter(cum_mxy_rotate[0], cum_mxy_rotate[1], marker='o', s=10, color="yellow", zorder=2)

        if show_start:
            # Plot the start
            axes.scatter(cum_mxy_rotate[0, 0], cum_mxy_rotate[1, 0], marker='o', s=100, color="purple", zorder=3,
                         alpha=0.5)

        if show_stops:
            zerospeed_index = np.where(cum_speeddummy == 0)
            zerospeedx, zerospeedy = cum_mx[zerospeed_index], cum_my[zerospeed_index]
            zerospeed_rotate = np.matmul(R, np.stack((zerospeedx, zerospeedy)))
            axes.scatter(zerospeed_rotate[0], zerospeed_rotate[1], marker='*', s=150, color="black", zorder=2)

        ff_position_rotate = np.matmul(R, np.stack(
            (ff_position_during_this_trial.T[0], ff_position_during_this_trial.T[1])))
        axes.scatter(ff_position_rotate[0], ff_position_rotate[1], marker='o', s=10, color="magenta", zorder=2)

        if show_believed_target_positions:
            ff_believed_position_rotate = np.matmul(R, np.stack((ff_believed_position_sorted[
                                                                 currentTrial - num_trials + 1:currentTrial + 1].T[0],
                                                                 ff_believed_position_sorted[
                                                                 currentTrial - num_trials + 1:currentTrial + 1].T[1])))
            axes.scatter(ff_believed_position_rotate[0], ff_believed_position_rotate[1], marker='*', s=120, color="red",
                         alpha=0.75, zorder=2)

        if show_reward_boundary:
            for i in ff_position_rotate.T:
                circle2 = plt.Circle((i[0], i[1]), 25, facecolor='grey', edgecolor='orange', alpha=0.45, zorder=1)
                axes.add_patch(circle2)

        if trial_to_show_cluster != None:
            # Find the indices of ffs in the cluster
            cluster_indices = np.unique(cluster_dataframe_point[cluster_dataframe_point[
                                                                    'target_index'] == currentTrial + trial_to_show_cluster].ff_index.to_numpy())
            cluster_ff_positions = ff_real_position_sorted[np.array(cluster_indices)]
            cluster_ff_rotate = np.matmul(R, np.stack((cluster_ff_positions.T[0], cluster_ff_positions.T[1])))
            axes.scatter(cluster_ff_rotate[0], cluster_ff_rotate[1], marker='o', c="blue", s=25, zorder=4)

        if show_connect_path_target.any():
            xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_target)]
            xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index'] == currentTrial) & (
                        xy_onoff_lines['visible'] == 1) & (xy_onoff_lines['ff_distance'] < 250)][
                                          ['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=30, c="green", alpha=0.4, zorder=4)

        if show_connect_path_pre_target.any():
            xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'].isin(show_connect_path_pre_target)]
            xy_onoff_lines = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index'] == currentTrial - 1) & (
                        xy_onoff_lines['visible'] == 1) & (xy_onoff_lines['ff_distance'] < 250)][
                                          ['monkey_x', 'monkey_y']])
            onoff_lines_rotate = np.matmul(R, xy_onoff_lines.T)
            axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=40, c="aqua", alpha=0.6, zorder=3)

        if show_connect_path_ff.any():
            target_nums = np.arange(currentTrial - num_trials + 1, currentTrial + 1)
            temp_dataframe = ff_dataframe.loc[ff_dataframe['target_index'].isin(target_nums)]
            temp_dataframe = temp_dataframe.loc[
                (temp_dataframe['ff_distance'] < 250) & (temp_dataframe['visible'] == 1)]
            temp_dataframe = temp_dataframe.loc[~temp_dataframe['ff_index'].isin(target_nums)][
                ['ff_x', 'ff_y', 'monkey_x', 'monkey_y']]
            temp_array = temp_dataframe.to_numpy()
            temp_ff_positions = np.matmul(R, temp_array[:, :2].T)
            temp_monkey_positions = np.matmul(R, temp_array[:, 2:].T)
            for j in range(len(temp_array)):
                axes.plot(np.stack([temp_ff_positions[0, j], temp_monkey_positions[0, j]]),
                          np.stack([temp_ff_positions[1, j], temp_monkey_positions[1, j]]), '-', alpha=0.2, c="#a940f5")
                axes.plot(temp_ff_positions[0, j], temp_ff_positions[1, j], '-', alpha=0.2, marker="o", markersize=5,
                          color="brown")

        # if show_connect_path_ff.any():
        #   if num_trials == 1:
        #     x, y = connect_path_ff(cum_t, cum_mx, cum_my, cum_angle, currentTrial, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
        #   else:
        #     target_nums = np.arange(currentTrial-num_trials+1, currentTrial+1)
        #     x, y = connect_path_ff2(cum_t, cum_mx, cum_my, cum_angle, target_nums, 250, ff_flash_sorted, ff_real_position_sorted, total_ff_num)
        #   for j in range(len(x)):
        #     xy_rotate = np.matmul(R, np.stack((x[j], y[j])))
        #     axes.plot(xy_rotate[0], xy_rotate[1], '-', alpha=0.2, c="#a940f5")
        #     axes.plot(xy_rotate[0][0], xy_rotate[1][0], '-', alpha=0.2, marker="o", markersize=5, color="brown")

        if trial_to_show_cluster_around_target != None:
            cluster_ff_pos = ffs_around_target_positions[currentTrial + trial_to_show_cluster_around_target]
            if len(cluster_ff_pos) > 0:
                ffs_around_target_rotate = np.matmul(R, np.stack((cluster_ff_pos.T[0], cluster_ff_pos.T[1])))
                axes.scatter(ffs_around_target_rotate[0], ffs_around_target_rotate[1], marker='o', s=30, color="blue",
                             zorder=4)
            if cluster_on_off_lines:
                # Find on_off_lines for ffs in the cluster
                for i in range(len(cluster_ff_pos)):
                    index = np.array(ff_dataframe[(np.isclose(np.array(ff_dataframe['ff_x']), cluster_ff_pos[i, 0])) & (
                        np.isclose(np.array(ff_dataframe['ff_y']), cluster_ff_pos[i, 1]))]['ff_index'])
                    if len(index) > 0:
                        index = index[0]
                        # index = ffs_around_target_indices[currentTrial-trial_to_show_cluster_around_target][i]
                        xy_onoff_lines = ff_dataframe.loc[ff_dataframe['target_index'] == currentTrial]
                        xy_onoff_lines2 = np.array(xy_onoff_lines.loc[(xy_onoff_lines['ff_index'] == index) & (
                                    xy_onoff_lines['visible'] == 1)][['monkey_x', 'monkey_y']])
                        onoff_lines_rotate = np.matmul(R, xy_onoff_lines2.T)
                        axes.scatter(onoff_lines_rotate[0], onoff_lines_rotate[1], s=15 - 3 * i,
                                     color=list_of_colors[i], alpha=0.4, zorder=3 + i)
                        # Use corresponding color for that ff
                        xy_onoff_lines3 = np.array(xy_onoff_lines[['ff_x', 'ff_y']])
                        ffs_around_target_rotate = np.matmul(R, xy_onoff_lines3.T)
                        axes.scatter(ffs_around_target_rotate[0], ffs_around_target_rotate[1], marker='o', s=100,
                                     alpha=0.5, color=list_of_colors[i], zorder=3)

        if show_scale_bar:
            scale1 = ScaleBar(
                dx=1, length_fraction=0.2, fixed_value=100,
                location='upper left',  # in relation to the whole plot
                label_loc='left', scale_loc='bottom'  # in relation to the line
            )
            axes.add_artist(scale1)
            axes.xaxis.set_major_locator(mtick.NullLocator())
            axes.yaxis.set_major_locator(mtick.NullLocator())

        xmin, xmax = np.min(cum_mxy_rotate[0]), np.max(cum_mxy_rotate[0])
        ymin, ymax = np.min(cum_mxy_rotate[1]), np.max(cum_mxy_rotate[1])
        bigger_width = max(xmax - xmin, ymax - ymin)
        xmiddle, ymiddle = (xmin + xmax) / 2, (ymin + ymax) / 2
        xmin, xmax = xmiddle - bigger_width / 2, xmiddle + bigger_width / 2
        ymin, ymax = ymiddle - bigger_width / 2, ymiddle + bigger_width / 2
        margin = max(bigger_width / 5, 150)
        if zoom_in == True:
            axes.set_xlim((xmin - 40, xmax + 40))
            axes.set_ylim((ymin - 20, ymax + 60))
        else:
            axes.set_xlim((xmin - margin, xmax + margin))
            axes.set_ylim((ymin - margin, ymax + margin))
        axes.set_aspect('equal')


        if show_colorbar == True:
            # Make the black and red colorbar
            A = np.reshape([1, 2, 1, 2, 2, 2], (2, 3))  # The numbers don't matter much
            norm_bins = np.array([0.5, 1.5, 2.5])
            # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
            col_dict = {1: "black", 2: "red"}
            # We create a colormar from our list of colors
            speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
            ## Make normalizer and formatter
            norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
            labels = np.array(["No Reward", "Reward"])
            fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
            # Plot our figure
            im = axes.imshow(A, cmap=speed_cm, extent=[0, 0, 0, 0], norm=norm)
            cax2 = fig.add_axes([0.95, 0.15, 0.05, 0.2])
            cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
            cb.ax.tick_params(width=0)
            cb.ax.set_title('Stopping Points', ha='left')
            if trail_color == "orange":
                A = np.reshape([1, 2, 1, 2, 2, 2], (2, 3))  # The numbers don't matter much
                norm_bins = np.array([0.5, 1.5, 2.5])
                # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
                col_dict = {1: "green", 2: "orange"}
                # We create a colormar from our list of colors
                speed_cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
                ## Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, 2, clip=True)
                labels = np.array(["Top Target Visible", "Top Target Not Visible"])
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
                # Plot our figure
                im = axes.imshow(A, cmap=speed_cm, extent=[0, 0, 0, 0], norm=norm)
                cax2 = fig.add_axes([0.95, 0.5, 0.05, 0.2])
                cb = fig.colorbar(im, format=fmt, ticks=np.array([1., 2.]), cax=cax2)
                cb.ax.tick_params(width=0)
                cb.ax.set_title('Path Colors', ha='left', y=1.04)
            elif trail_color == "viridis":
                cmap = cm.viridis
                norm = matplotlib.colors.Normalize(vmin=0, vmax=200)
                cax = fig.add_axes([0.95, 0.4, 0.05, 0.43])
                cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                    cax=cax, orientation='vertical')
                cbar.ax.set_title('Speed(cm/s)', ha='left', y=1.04)
                cbar.ax.tick_params(axis='y', color='white', direction="in", right=True, length=5, width=1.5)
                cbar.outline.remove()


"""### PlotPoints"""


def PlotPoints(cum_mx, cum_my, ff_position_during_this_trial, ff_real_position_sorted, ff_believed_position_sorted,
               cluster_dataframe_point, ff_dataframe,
               point,
               total_time,
               show_all_ff,
               show_flash_on_ff,
               show_visible_ff,
               show_in_memory_ff,
               show_target,
               show_reward_boundary,
               show_legend,
               show_colorbar,
               show_scale_bar,
               trial_num=None,
               **kwargs):
    alive_ff_indices = np.array([i for i, value in
                                 enumerate(ff_life_sorted) if (value[-1] >= time) and (value[0] < time)])
    alive_ff_positions = ff_real_position_sorted[alive_ff_indices]
    if show_all_ff:
        axes.scatter(alive_ff_positions.T[0], alive_ff_positions.T[1], color="grey", s=30)

    if show_flash_on_ff:
        on_ff_indices = []  # Gives the indices of the ffs that are on at this point
        # For each firefly in ff_flash_sorted:
        for index, firefly in enumerate(ff_flash_sorted):
            # If the firefly has flashed during that trial:
            if index in alive_ff_indices:
                # Let's see if the firefly has flashed at that exact moment
                for interval in firefly:
                    if interval[0] <= time <= interval[1]:
                        on_ff_indices.append(index)
                        break
        on_ff_indices = np.array(on_ff_indices)
        on_ff_positions = ff_real_position_sorted[on_ff_indices]
        axes.scatter(on_ff_positions.T[0], on_ff_positions.T[1], color="red", s=120, marker='*', alpha=0.7)

    if show_visible_ff:
        visible_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 1)][
            ['ff_x', 'ff_y']]
        axes.scatter(visible_ffs['ff_x'], visible_ffs['ff_y'], color="orange", s=40)

    if show_in_memory_ff:
        in_memory_ffs = ff_dataframe[(ff_dataframe['point_index'] == point) & (ff_dataframe['visible'] == 0)][
            ['ff_x', 'ff_y']]
        axes.scatter(in_memory_ffs['ff_x'], in_memory_ffs['ff_y'], color="green", s=40)

    if show_target:
        if trial_num == None:
            raise ValueError("If show_target, then trial_num cannot be None")
        target_num = distance_dataframe['trial'].iloc[point]
        target_position = ff_real_position_sorted[trial_num]
        axes.scatter(target_position[0], target_position[1], marker='*', s=200, color="grey", alpha=0.35)

    if show_legend == True:
        legend_names = []
        if show_all_ff:
            legend_names.append("Invisible")
        if show_flash_on_ff:
            legend_names.append("Flash On")
        if show_visible_ff:
            legend_names.append("Visible")
        if show_in_memory_ff:
            legend_names.append("In memory")
        if show_target:
            legend_names.append("Target")
        axes.legend(legend_names, loc='upper right')
        global Show_Legend
        Show_Legend = False

    if show_reward_boundary:
        if show_all_ff:
            for i in range(len(alive_ff_positions)):
                circle2 = plt.Circle((alive_ff_positions[i, 0], alive_ff_positions[i, 1]), 20, facecolor='grey',
                                     edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle2)
        elif show_flash_on_ff:
            if show_flash_on_ff:
                for i in range(len(on_ff_positions)):
                    circle2 = plt.Circle((on_ff_positions[i, 0], on_ff_positions[i, 1]), 20, facecolor='grey',
                                         edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle2)
            if show_in_memory_ff:
                for i in range(len(in_memory_ffs)):
                    circle2 = plt.Circle((in_memory_ffs['ff_x'].iloc[i], in_memory_ffs['ff_y'].iloc[i]), 20,
                                         facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle2)
        elif show_visible_ff:
            for i in range(len(visible_ffs)):
                circle2 = plt.Circle((visible_ffs['ff_x'].iloc[i], visible_ffs['ff_y'].iloc[i]), 20, facecolor='grey',
                                     edgecolor='orange', alpha=0.25, zorder=1)
                axes.add_patch(circle2)
            if show_in_memory_ff:
                for i in range(len(in_memory_ffs)):
                    circle2 = plt.Circle((in_memory_ffs['ff_x'].iloc[i], in_memory_ffs['ff_y'].iloc[i]), 20,
                                         facecolor='grey', edgecolor='orange', alpha=0.25, zorder=1)
                    axes.add_patch(circle2)

    axes.scatter(cum_mx, cum_my, s=15, c=index_temp, cmap="Blues")

    xmin, xmax = np.min(cum_mx), np.max(cum_mx)
    ymin, ymax = np.min(cum_my), np.max(cum_my)
    bigger_width = max(xmax - xmin, ymax - ymin)
    xmiddle, ymiddle = (xmin + xmax) / 2, (ymin + ymax) / 2
    xmin, xmax = xmiddle - bigger_width / 2, xmiddle + bigger_width / 2
    ymin, ymax = ymiddle - bigger_width / 2, ymiddle + bigger_width / 2
    margin = max(bigger_width / 5, 250)
    axes.set_xlim((xmin - margin, xmax + margin))
    axes.set_ylim((ymin - margin, ymax + margin))
    axes.set_aspect('equal')

    if show_scale_bar == True:
        scale1 = ScaleBar(
            dx=1, length_fraction=0.2, fixed_value=100,
            location='upper left',  # in relation to the whole plot
            label_loc='left', scale_loc='bottom'  # in relation to the line
        )
        axes.add_artist(scale1)

    axes.xaxis.set_major_locator(mtick.NullLocator())
    axes.yaxis.set_major_locator(mtick.NullLocator())

    if show_colorbar == True:
        cmap = cm.Blues
        cax = fig.add_axes([0.95, 0.25, 0.05, 0.52])  # [left, bottom, width, height]
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ticks=[0, 1],
                            cax=cax, orientation='vertical')
        cbar.ax.set_title('Trajectory', ha='left', y=1.07)
        cbar.ax.tick_params(size=0)
        cbar.outline.remove()
        cbar.ax.set_yticklabels(['Least recent', 'Most recent'])
        global Show_Colorbar
        Show_Colorbar = False