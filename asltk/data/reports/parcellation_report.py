# import os
# from datetime import datetime

# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages

# from asltk import PARCELLATION_REPORT_PATH as default_path
# from asltk.asldata import ASLData
# from asltk.data.brain_atlas import BrainAtlas
# from asltk.data.reports.basic_report import BasicReport
# from asltk.utils.io import load_image


# class ParcellationReport(BasicReport):
#     def __init__(
#         self,
#         subject_image: ASLData,
#         atlas_name: str = 'MNI2009',
#         subject_filename: str = None,
#         subject_img_dimensions: tuple = None,
#         subject_img_type: str = None,
#         subject_img_resolution: tuple = None,
#         **kwargs,
#     ):
#         self.atlas = load_image(BrainAtlas(atlas_name).get_atlas()['t1_data'])
#         self.subject_image = subject_image('m0')
#         self._check_inputs_dimensions(self.subject_image, self.atlas)

#         # Optional parameters for subject information
#         self.subject_filename = (
#             subject_filename if subject_filename else 'Unknown'
#         )
#         self.subject_img_dimensions = (
#             subject_img_dimensions if subject_img_dimensions else (0, 0, 0)
#         )
#         self.subject_img_type = (
#             subject_img_type if subject_img_type else 'Unknown'
#         )
#         self.subject_img_resolution = (
#             subject_img_resolution if subject_img_resolution else (0, 0, 0)
#         )

#         default_filename = f"parcellation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
#         self.report_filename = kwargs.get('report_filename', default_filename)

#         self.default_fullpath = os.path.join(
#             default_path, self.report_filename
#         )

#         # Initialize the report data structure
#         self.report_data = {}

#     def generate_report(self):
#         # Report structure:
#         # Description section:
#         # - Report information: date
#         # - Brain Atlas: Name and description
#         # - Brain Regions: List of regions with their labels and descriptions
#         # - Subject Information: Subject filename, image dimensions, image type, image resolution
#         # Illustration section:
#         # - Brain atlas illustration: Image of the brain atlas with regions labeled (5 slices I-S)
#         # - Subject illustration: Image of subject's brain without parcellation (5 slices I-S)
#         # - Subject illustration: Image of the subject's brain with parcellation overlay (5 slices I-S)
#         # Parcellation section:
#         # - Table with parcellation statistics:
#         #   - Region label
#         #   - Region name
#         #   - Number of voxels
#         #   - Volume in mm³
#         #   - Average intensity
#         #   - Std. deviation of intensity
#         #   - Minimum intensity
#         #   - Maximum intensity
#         #   - Coefficient of variation (CV)
#         description_section = self._create_description_section()

#         self.report_data = description_section

#     def save_report(self, format: str = 'csv'):
#         # TODO explain in the documentation that the file path is defined by the report_filename and uses the PARCELLATION_REPORT_PATH in the asltk module
#         if not self.report_data:
#             raise ValueError(
#                 'Report data is empty. Please generate the report first.'
#             )

#         # Save the report data to a file
#         if format == 'csv':
#             # TODO revise the CSV formatting to include all necessary information
#             # Save the regions DataFrame to a CSV file
#             self.report_data['regions_dataframe'].to_csv(
#                 self.default_fullpath, index=False
#             )
#         elif format == 'pdf':
#             # Save the report as a PDF file
#             with PdfPages(self.default_fullpath) as pdf:
#                 # Save the header figure
#                 pdf.savefig(self.report_data['header_figure'])
#                 plt.close(self.report_data['header_figure'])

#                 # Add more sections to the PDF as needed
#                 # For example, you can add illustrations or parcellation statistics here

#     def _create_description_section(self):
#         """
#         Create the description section header for the PDF report.

#         Returns:
#             dict: A dictionary containing the matplotlib figures and information for the report header.
#         """

#         # Create figure for the header section
#         fig = plt.figure(figsize=(10, 8))
#         gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 2])

#         # Report information: date
#         ax1 = plt.subplot(gs[0])
#         ax1.axis('off')
#         ax1.text(
#             0.01, 0.5, f'Parcellation Report', fontsize=16, fontweight='bold'
#         )
#         ax1.text(
#             0.01,
#             0.1,
#             f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
#             fontsize=10,
#         )

#         # Brain Atlas: Name and description
#         ax2 = plt.subplot(gs[1])
#         ax2.axis('off')
#         ax2.text(
#             0.01,
#             0.7,
#             f'Brain Atlas Information',
#             fontsize=14,
#             fontweight='bold',
#         )
#         ax2.text(0.01, 0.4, f'Name: {self.atlas.name}')
#         ax2.text(
#             0.01,
#             0.1,
#             f"Description: {getattr(self.atlas, 'description', 'No description available')}",
#         )

#         # Subject Information
#         ax3 = plt.subplot(gs[2])
#         ax3.axis('off')
#         ax3.text(
#             0.01, 0.9, 'Subject Information', fontsize=14, fontweight='bold'
#         )
#         ax3.text(0.01, 0.7, f'Filename: {self.subject_filename}')
#         ax3.text(0.01, 0.5, f'Image dimensions: {self.subject_img_dimensions}')
#         ax3.text(0.01, 0.3, f'Image type: {self.subject_img_type}')
#         ax3.text(
#             0.01, 0.1, f'Image resolution: {self.subject_img_resolution} mm'
#         )

#         # Brain Regions: Create a DataFrame with the regions information
#         try:
#             regions_data = {'Label': [], 'Region Name': []}

#             # Get regions from the atlas - adapt this based on how your BrainAtlas class works
#             for label, region in self.atlas.get('labels', {}).items():
#                 regions_data['Label'].append(label)
#                 regions_data['Region Name'].append(region)
#                 # regions_data['Description'].append(getattr(region, 'description', 'No description available'))

#             df_regions = pd.DataFrame(regions_data)

#             # Create a table for the regions
#             ax4 = plt.subplot(gs[3])
#             ax4.axis('off')
#             ax4.text(
#                 0.01, 0.95, 'Brain Regions', fontsize=14, fontweight='bold'
#             )

#             # Display all regions in a table
#             table_data = df_regions.values
#             columns = df_regions.columns

#             table = ax4.table(
#                 cellText=table_data,
#                 colLabels=columns,
#                 loc='center',
#                 cellLoc='center',
#                 colWidths=[0.1, 0.3, 0.6],
#             )
#             table.auto_set_font_size(False)
#             table.set_fontsize(8)
#             table.scale(1, 1.5)

#         except Exception as e:
#             # In case of any error with regions
#             ax4 = plt.subplot(gs[3])
#             ax4.axis('off')
#             ax4.text(
#                 0.01,
#                 0.5,
#                 f'Brain Regions: Error retrieving region information. {str(e)}',
#                 fontsize=10,
#                 color='red',
#             )
#             df_regions = pd.DataFrame()

#         plt.tight_layout()

#         # Return the result as a dictionary that can be used by save_report
#         return {
#             'header_figure': fig,
#             'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
#             'atlas_name': self.atlas.get('atlas_name', 'Unknown Atlas'),
#             'atlas_description': self.atlas.get(
#                 'description', 'No description available'
#             ),
#             'subject_info': {
#                 'filename': self.subject_filename,
#                 'dimensions': self.subject_img_dimensions,
#                 'type': self.subject_img_type,
#                 'resolution': self.subject_img_resolution,
#             },
#             'regions_dataframe': df_regions,
#         }

#     def _check_inputs_dimensions(subject_image, atlas):
#         subj_dim = subject_image.shape
#         atlas_dim = atlas.shape
#         if subj_dim != atlas_dim:
#             raise TypeError(
#                 f'subject_image must have the same dimensions as the atlas image. Dimensions do not match: {subj_dim} != {atlas_dim}'
#             )
