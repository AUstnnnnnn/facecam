# dmgbuild settings for Facecam.dmg: drag-to-Applications window. build.sh runs this.
# dmg-bg@2x.png next to dmg-bg.png is picked up for Retina screens.
files = ["dist/Facecam.app"]
symlinks = {"Applications": "/Applications"}
icon = "dist/Facecam.app/Contents/Resources/Facecam.icns"
background = "dmg-bg.png"
window_rect = ((200, 120), (640, 400))
icon_size = 128
text_size = 13
icon_locations = {"Facecam.app": (160, 190), "Applications": (480, 190)}
show_status_bar = show_tab_view = show_toolbar = show_pathbar = show_sidebar = False
default_view = "icon-view"
format = "UDZO"
