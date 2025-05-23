#include "openloong_br.h"

int openloong_br::scanKeyboard()
{
    tcgetattr(0,&stored_settings);
    new_settings = stored_settings;
    new_settings.c_lflag &= (~ICANON);
    new_settings.c_cc[VTIME] = 0;
    tcgetattr(0,&stored_settings);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0,TCSANOW,&new_settings);
    in_scanKeyboard = getchar();
    tcsetattr(0,TCSANOW,&stored_settings);
    return in_scanKeyboard;
}

int openloong_br::updateKey(void) {
    key_updateKey =scanKeyboard();
    if (key_updateKey > 0) {
        return key_updateKey;
    }
    else
        return -1;
}
