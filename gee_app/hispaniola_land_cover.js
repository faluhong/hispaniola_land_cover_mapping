// ### GEE code to display the land cover in Hispaniola (Haiti and the Dominican Republic) region.
// ### Corresponding to GEE app link is: https://gers.users.earthengine.app/view/hispaniola-lc

// variable to store all the annual land cover
var landCover = {
    'Google Earth Image': getLandCoverMap('2022'), // 'Google Earth Image
    1996: getLandCoverMap('1996'),
    1997: getLandCoverMap('1997'),
    1998: getLandCoverMap('1998'),
    1999: getLandCoverMap('1999'),
    2000: getLandCoverMap('2000'),
    2001: getLandCoverMap('2001'),
    2002: getLandCoverMap('2002'),
    2003: getLandCoverMap('2003'),
    2004: getLandCoverMap('2004'),
    2005: getLandCoverMap('2005'),
    2006: getLandCoverMap('2006'),
    2007: getLandCoverMap('2007'),
    2008: getLandCoverMap('2008'),
    2009: getLandCoverMap('2009'),
    2010: getLandCoverMap('2010'),
    2011: getLandCoverMap('2011'),
    2012: getLandCoverMap('2012'),
    2013: getLandCoverMap('2013'),
    2014: getLandCoverMap('2014'),
    2015: getLandCoverMap('2015'),
    2016: getLandCoverMap('2016'),
    2017: getLandCoverMap('2017'),
    2018: getLandCoverMap('2018'),
    2019: getLandCoverMap('2019'),
    2020: getLandCoverMap('2020'),
    2021: getLandCoverMap('2021'),
    2022: getLandCoverMap('2022'),
};


var paletteLandCover = [
    'f10100', // developed
    '1d6533', // primary wet forest
    'd0d181', // primary dry forest
    '6ca966', // secondary forest
    'ae7229', // shrub/grass
    '486da2', // water
    'c8e6f8', // wetland
    'b3afa4', // other
];
var visParaCoverTypes = {min: 1, max: 8, palette: paletteLandCover};

var landCoverLegend = [
    {'Developed': 'f10100'},
    {'Primary wet forest': '1d6533'},
    {'Primary dry forest': 'd0d181'},
    {'Secondary forest': '6ca966'},
    {'Shrub/Grass': 'ae7229'},
    {'Water': '486da2'},
    {'Wetland': 'c8e6f8'},
    {'Other': 'b3afa4'},
];

var shapefile = ee.FeatureCollection('users/gers/Hispaniola/hispaniola_polygon');
var styling = {
    color: 'black',
    width: 1.5,
    fillColor: '#00000000',   // eight-digit hex code, last two codes represent opacity
};  //Define styling and determine the color of the shapefile
shapefile = shapefile.style(styling);

var leftMap = ui.Map();
leftMap.setControlVisibility(false);
var leftTitle = ui.Label('Left Map Title', {fontWeight: 'bold', fontSize: '24px', position: 'top-left'});
leftMap.add(leftTitle);

// addRefMapSelector(leftMap, 0, 'bottom-left', leftTitle);
addLandCoverSelector(leftMap, 26, 'top-left', leftTitle);   // set the default display year as 2022
leftMap.addLayer(shapefile, {}, 'Country Boundaries');   // add the country boundary shapefile

var rightMap = ui.Map();
rightMap.setControlVisibility(false);
var rightTitle = ui.Label('Right Map Title', {fontWeight: 'bold', fontSize: '24px', position: 'top-right'});
rightMap.add(rightTitle);

addLandCoverSelector(rightMap, 27, 'top-right', rightTitle);
// addRefMapSelector(rightMap, 0, 'bottom-right', rightTitle);
rightMap.addLayer(shapefile, {}, 'Country Boundaries');  // add the country boundary shapefile


var splitPanel = ui.SplitPanel({
    firstPanel: leftMap,
    secondPanel: rightMap,
    wipe: true,
    style: {stretch: 'both'}
});

// Set the SplitPanel as the only thing in the UI root.
ui.root.widgets().reset([splitPanel]);
var linker = ui.Map.Linker([leftMap, rightMap]);

leftMap.setCenter(-70.96996964, 19.02679734, 12); //default location
leftMap.setOptions('SATELLITE'); //Set satellite as the base layer
rightMap.setCenter(-70.96996964, 19.02679734, 12);
rightMap.setOptions('SATELLITE');


// add panels
var verticalFlow = ui.Panel.Layout.flow('vertical');

var header = ui.Label('Hispaniola land cover map', {fontSize: '20px', color: 'Green', fontWeight: 'bold'});
var text = ui.Label(
    'Annual 30-meter land cover map including the primary and secondary forests in Haiti and the Dominican Republic from 1996 to 2022.',
    {fontSize: '12px'});
var toolPanel = ui.Panel([header, text], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
ui.root.widgets().add(toolPanel);


// define a panel for the legend of cover map
var legendCoverPanel = ui.Panel({
    style:
        {fontWeight: 'bold', fontSize: '12px', margin: '0 0 0 8px', padding: '0'}
});
toolPanel.add(legendCoverPanel);

var legendCoverTitle = ui.Label(
    'Legend',
    {fontWeight: 'bold', fontSize: '12px', margin: '10px 0 4px 0', padding: '0'});
legendCoverPanel.add(legendCoverTitle);

var keyCoverPanel = ui.Panel();
legendCoverPanel.add(keyCoverPanel);

setLegendLandCoverTypes(keyCoverPanel, landCoverLegend);  // Set the legend of cover map


// add an empty panel for spacing
var emptyLabel = ui.Label(' ', {fontSize: '12px'});
var emptyPanel = ui.Panel();
emptyPanel.add(emptyLabel);
toolPanel.add(emptyPanel);


// add the source code panel
var sourceCodeHeader = ui.Label('Source code', {fontSize: '12px', fontWeight: 'bold'});
var sourceCodeText = ui.Label(
    'GitHub: Hispaniola Land Cover Mapping',
    {fontSize: '12px'});
sourceCodeText.setUrl('https://github.com/faluhong/hispaniola_land_cover_mapping');
var sourceCodePanel = ui.Panel([sourceCodeHeader, sourceCodeText], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(sourceCodePanel);


// add the data download panel
var sourceDataDownloadHeader = ui.Label('Data download', {fontSize: '12px', fontWeight: 'bold'});
var sourceDataDownloadText = ui.Label(
    'https://doi.org/10.6084/m9.figshare.28100408',
    {fontSize: '12px'});
sourceDataDownloadText.setUrl('https://doi.org/10.6084/m9.figshare.28100408');
var sourceDataDownload = ui.Panel([sourceDataDownloadHeader, sourceDataDownloadText], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(sourceDataDownload);



// add the reference paper panel
var sourceRefPaperHeader = ui.Label('Reference paper', {fontSize: '12px', fontWeight: 'bold'});
var sourceRefPaperText = ui.Label(
    'Falu Hong, S. Blair Hedges, Zhiqiang Yang, Ji Won Suh, Shi Qiu, Joel Timyan, and Zhe Zhu. Decoding primary forest changes in Haiti and the Dominican Republic using Landsat time series. Remote Sensing of Environment, 318, 114590',
    {fontSize: '12px'});
sourceRefPaperText.setUrl('https://www.sciencedirect.com/science/article/pii/S0034425724006163');
var sourceRefPaper = ui.Panel([sourceRefPaperHeader, sourceRefPaperText], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(sourceRefPaper);


// add contact panel
var contactHeader = ui.Label('Contact', {fontSize: '12px', fontWeight: 'bold'});
var contactText = ui.Label(
    'Falu Hong: faluhong@uconn.edu',
    {fontSize: '12px'});
var contactPanel = ui.Panel([contactHeader, contactText], 'flow', {fontWeight: 'bold', fontSize: '12px', width: '300px'});
toolPanel.add(contactPanel);


// add an empty panel for spacing
var emptyPanel = ui.Panel();
toolPanel.add(emptyPanel);


// add logos to the panel
var logo = ee.Image('users/gers/Hispaniola/logo').visualize({
    bands: ['b1', 'b2', 'b3'],
    min: 0,
    max: 255
});
var thumb = ui.Thumbnail({
    image: logo,
    params: {
        dimensions: '2683x1080',
        format: 'png'
    },
    style: {height: '109px', width: '270px', padding: '10'}
});

toolPanel.add(thumb);

// set legend for the land cover types
function setLegendLandCoverTypes(setPanel, legendTypes) {
    for (var i = 0; i < legendTypes.length; i++) {
        var item = legendTypes[i];
        var name = Object.keys(item)[0];
        var color = item[name];
        var colorBox = ui.Label('', {
            backgroundColor: color,
            // Use padding to give the box height and width.
            padding: '9px',
            margin: '0'
        });
        // Create the label with the description text.
        var description = ui.Label(name, {margin: '0 0 4px 6px'});
        setPanel.add(
            ui.Panel([colorBox, description], ui.Panel.Layout.Flow('horizontal')));
    }
}


// Adds a layer selection widget to the given year, to allow users to display the land cover on the given year in the associated map
function addLandCoverSelector(mapToChange, defaultValue, position, title) {
    var label = ui.Label('Annual Land Cover', {fontWeight: 'bold', fontSize: '12px'});

    function updateMap(selection) {

        if (ee.String(selection).compareTo('Google Earth Image').getInfo() === 0) {
            mapToChange.layers().set(0);  // Display the default Google Earth Image
            title.setValue('Google Earth Image');
        } else {
            var landCoverSelection = landCover[selection].visualize(visParaCoverTypes);

            mapToChange.layers().set(0, ui.Map.Layer(landCoverSelection));
            title.setValue('Hispaniola land cover in ' + selection);
        }

    }

    var select = ui.Select({items: Object.keys(landCover), onChange: updateMap});
    select.setValue(Object.keys(landCover)[defaultValue], true);

    var controlPanel = ui.Panel({widgets: [label, select], style: {position: position}});
    mapToChange.add(controlPanel);
}


// read the land cover map from the Google Earth Engine
function getLandCoverMap(year) {
    var filenameImage = 'users/gers/Hispaniola/hispaniola_lc_' + year;
    var landCover = ee.Image(filenameImage);
    landCover = landCover.updateMask(landCover.neq(0));  // mask out the ocean pixels

    return landCover;
}
