// create the slippy map
var map = L.map('map', {
	minZoom: 1,
	maxZoom: 6,
	center: [0, 0],
	zoom: 2,
	crs: L.CRS.Simple,
    });

// dimensions of the image
//var w = 4000,
//    h = 4000,
//    url = 'http://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_4k.jpg';
//var w = 6000,
//    h = 6000,
//    url = 'http://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_6k.jpg';
var w = 6000,
    h = 4000,
    url = 'http://stanford.edu/~rkeisler/viz/test.jpg';

// calculate the edges of the image, in coordinate space
var southWest = map.unproject([0, h], map.getMaxZoom()-1);
var northEast = map.unproject([w, 0], map.getMaxZoom()-1);
var bounds = new L.LatLngBounds(southWest, northEast);

// add the image overlay, 
// so that it covers the entire map
L.imageOverlay(url, bounds).addTo(map);

// tell leaflet that the map is exactly as big as the image
map.setMaxBounds(bounds);

