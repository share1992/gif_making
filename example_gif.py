import gif_making as gm

pca = []
irc_path = "example_files/mp2_631pgdp_energies_plus_correct_irc_points.txt"
output_path = "."

gm.plot_irc_for_gif(irc_path, gif_points_to_circle="all", imgname="bifurcation_plot", output_directory=output_path)
