class Main {
    public static void main(String[] args) {

        Manage m = new Manage();
        Drone d = new Drone();
        Cab c = new Cab();
        Bike b = new Bike();

        m.manage(d);
        m.manage(c);
        m.manage(b);

    }
}