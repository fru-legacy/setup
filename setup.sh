echo '\n - Blacklist audio driver'
# Backlist kernel audio module (List with: lsmod | grep -i snd)
sudo touch /etc/modprobe.d/snd-blacklist.conf
cd /etc/modprobe.d/
echo 'blacklist soundcore'     | sudo tee --append blacklist.conf
echo 'blacklist snd'           | sudo tee --append blacklist.conf
echo 'blacklist snd_hda_intel' | sudo tee --append blacklist.conf
echo 'blacklist snd_hda_codec' | sudo tee --append blacklist.conf
echo 'blacklist snd_hda_core'  | sudo tee --append blacklist.conf

echo '\n - Install gdrive'
sudo apt-add-repository 'deb http://shaggytwodope.github.io/repo ./'
sudo apt-get update 
sudo apt-get install drive

echo '\n - Get install folder'
cd ~
drive init
drive pull -no-prompt install/packages

echo '\n - Install local packages'
sudo dpkg -i ~/install/packages/*.deb
sudo apt-get -f install

echo '\n - Install repository packages'
sudo apt-get install git
sudo apt-get install python3-pip python3-setuptools

echo '\n - Install docker compose'
sudo pip3 install --upgrade pip
sudo pip3 install nvidia-docker-compose
sudo apt-get install nvidia-modprobe

curl -L https://github.com/docker/machine/releases/download/v0.12.2/docker-machine-`uname -s`-`uname -m` >/tmp/docker-machine
chmod +x /tmp/docker-machine
sudo cp /tmp/docker-machine /usr/local/bin/docker-machine
sudo usermod -a -G docker $USER

echo '\n - Install node js'
curl -sL https://deb.nodesource.com/setup_6.x | sudo -E bash -
sudo apt-get install -y nodejs

echo '\n - Install inkscape and extensions'
sudo apt-get install inkscape
git clone https://github.com/ChristianBecker/inkscape-android-export.git
cp inkscape-android-export/android_export.inx ~/.config/inkscape/extensions
cp inkscape-android-export/android_export.py ~/.config/inkscape/extensions
sudo rm -rf inkscape-android-export

echo '\n - Equation editor'
sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended

echo '\n - Add links'
mv ~/install/desktop/anyform.txt ~/Schreibtisch/anyform.desktop
mv ~/install/desktop/react-multi-tree.txt ~/Schreibtisch/react-multi-tree.desktop
