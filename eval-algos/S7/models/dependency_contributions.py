# The output is a CSV file showing, for each devtooling project:
#    - The dependency (onchain project) id and name,
#    - The sum of weighted edge values (v_edge) contributed by that dependency,
#    - The fraction of the total package dependency contribution for that devtooling project.

# This is used to isolate weights of the algorithm used for dependency packages as seen int he file devtooling_openrank.py


import argparse
import numpy as np
import pandas as pd
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_snapshot):
    def get_path(filename):
        return f"{data_snapshot['data_dir']}/{filename}"

    df_onchain = pd.read_csv(get_path(data_snapshot['onchain_projects']))
    df_devtooling = pd.read_csv(get_path(data_snapshot['devtooling_projects']))
    df_dependencies = pd.read_csv(get_path(data_snapshot['project_dependencies']))
    df_devs2projects = pd.read_csv(get_path(data_snapshot['developers_to_projects']))
    df_devs2projects['event_month'] = pd.to_datetime(df_devs2projects['event_month'])
    return df_onchain, df_devtooling, df_dependencies, df_devs2projects

def build_dependency_graph(df_onchain, df_devtooling, df_dependencies, df_devs2projects):
    project_mapping = {
        **df_onchain.set_index('project_id')['display_name'].to_dict(),
        **df_devtooling.set_index('project_id')['display_name'].to_dict()
    }

    time_ref = df_devs2projects['event_month'].max()

    df_dep = df_dependencies.copy()
    df_dep.rename(
        columns={
            'onchain_builder_project_id': 'i',
            'devtooling_project_id': 'j',
            'dependency_source': 'event_type'
        },
        inplace=True
    )
    df_dep['event_month'] = time_ref  # use the same time_ref for all dependency events
    df_dep['i_name'] = df_dep['i'].map(project_mapping)
    df_dep['j_name'] = df_dep['j'].map(project_mapping)
    df_dep['link_type'] = 'PACKAGE_DEPENDENCY'
    return df_dep

def compute_weighted_edges(df_edges, config):
    df_edges = df_edges.copy()
    df_edges['event_type'] = df_edges['event_type'].str.upper()

    link_type_decay_factors = {k.upper(): v for k, v in config['simulation']['link_type_time_decays'].items()}
    event_type_weights = {k.upper(): v for k, v in config['simulation']['event_type_weights'].items()}

    decay_factor = link_type_decay_factors.get('PACKAGE_DEPENDENCY', 1.0)
    decay_lambda = np.log(2) / decay_factor

    time_ref = df_edges['event_month'].max()
    if not np.issubdtype(df_edges['event_month'].dtype, np.datetime64):
        df_edges['event_month'] = pd.to_datetime(df_edges['event_month'])
    time_diff_days = (time_ref - df_edges['event_month']).dt.days
    time_diff_years = time_diff_days / 365.0

    df_edges['v_edge'] = np.exp(-decay_lambda * time_diff_years) * df_edges['event_type'].map(event_type_weights)
    return df_edges

def compute_dependency_contributions(df_weighted):
    df_pkg = df_weighted[df_weighted['link_type'] == 'PACKAGE_DEPENDENCY'].copy()

    grouped = (
        df_pkg.groupby(['j', 'i', 'i_name', 'j_name'], as_index=False)
        ['v_edge']
        .sum()
    )
    total_per_project = grouped.groupby('j')['v_edge'].transform('sum')
    grouped['fraction'] = grouped['v_edge'] / total_per_project
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Compute per dependency contribution to final devtooling project scores."
    )
    parser.add_argument('config_file', nargs='?', default='eval-algos/S7/weights/devtooling_openrank_testing.yaml',
                        help="Path to YAML configuration file (default: eval-algos/S7/weights/devtooling_openrank_testing.yaml)")
    args = parser.parse_args()

    config = load_config(args.config_file)
    data_snapshot = config['data_snapshot']
    df_onchain, df_devtooling, df_dependencies, df_devs2projects = load_data(data_snapshot)

    df_dep_graph = build_dependency_graph(df_onchain, df_devtooling, df_dependencies, df_devs2projects)

    df_weighted = compute_weighted_edges(df_dep_graph, config)

    df_contrib = compute_dependency_contributions(df_weighted)

    print("Per Dependency Contributions (weighted v_edge and fraction):")
    print(df_contrib)

    output_path = f"{data_snapshot['data_dir']}/dependency_contributions.csv"
    df_contrib.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved dependency contributions to {output_path}")


if __name__ == "__main__":
    main()
